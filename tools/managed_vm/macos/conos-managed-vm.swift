import Darwin

let helperVersion = "conos.managed_vm_helper.macos/v0.1"

enum JSONValue {
    case string(String)
    case bool(Bool)
    case array([String])
}

func writeStdout(_ value: String) {
    value.withCString { ptr in
        _ = write(STDOUT_FILENO, ptr, strlen(ptr))
    }
}

func writeStderr(_ value: String) {
    value.withCString { ptr in
        _ = write(STDERR_FILENO, ptr, strlen(ptr))
    }
}

func eprint(_ value: String) {
    writeStderr(value + "\n")
}

func jsonEscape(_ value: String) -> String {
    var result = ""
    for scalar in value.unicodeScalars {
        switch scalar.value {
        case 34:
            result += "\\\""
        case 92:
            result += "\\\\"
        case 10:
            result += "\\n"
        case 13:
            result += "\\r"
        case 9:
            result += "\\t"
        default:
            result.append(Character(scalar))
        }
    }
    return result
}

func renderJSONValue(_ value: JSONValue) -> String {
    switch value {
    case .string(let text):
        return "\"\(jsonEscape(text))\""
    case .bool(let flag):
        return flag ? "true" : "false"
    case .array(let values):
        return "[" + values.map { "\"\(jsonEscape($0))\"" }.joined(separator: ", ") + "]"
    }
}

func jsonObject(_ values: [(String, JSONValue)]) -> String {
    let body = values.map { key, value in
        "  \"\(jsonEscape(key))\": \(renderJSONValue(value))"
    }.joined(separator: ",\n")
    return "{\n\(body)\n}"
}

func jsonLine(_ values: [(String, JSONValue)]) {
    writeStdout(jsonObject(values) + "\n")
}

func value(after flag: String, in args: [String], default defaultValue: String = "") -> String {
    guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else {
        return defaultValue
    }
    return args[idx + 1]
}

func commandAfterSeparator(_ args: [String]) -> [String] {
    guard let idx = args.firstIndex(of: "--"), idx + 1 < args.count else {
        return []
    }
    return Array(args[(idx + 1)...])
}

func expandUser(_ path: String) -> String {
    if path == "~" {
        return String(cString: getenv("HOME"))
    }
    if path.hasPrefix("~/") {
        return String(cString: getenv("HOME")) + String(path.dropFirst())
    }
    return path
}

func fileExists(_ path: String) -> Bool {
    return access(path, F_OK) == 0
}

func createSparseFile(_ path: String, sizeBytes: Int64) -> Bool {
    if fileExists(path) {
        return true
    }
    let fd = open(path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
    if fd < 0 {
        return false
    }
    defer {
        close(fd)
    }
    return ftruncate(fd, off_t(sizeBytes)) == 0
}

func ensureDir(_ path: String) -> Bool {
    let expanded = expandUser(path)
    var current = expanded.hasPrefix("/") ? "/" : ""
    for componentSub in expanded.split(separator: "/") {
        let component = String(componentSub)
        current = current == "/" || current.isEmpty ? current + component : current + "/" + component
        if mkdir(current, 0o755) != 0 && errno != EEXIST {
            return false
        }
    }
    return true
}

func virtualizationFrameworkAvailable() -> Bool {
    let path = "/System/Library/Frameworks/Virtualization.framework/Virtualization"
    guard let handle = dlopen(path, RTLD_LAZY) else {
        return false
    }
    dlclose(handle)
    return true
}

func providerPayload(args: [String]) -> [(String, JSONValue)] {
    let stateRoot = expandUser(value(after: "--state-root", in: args, default: "~/.conos/vm"))
    let imageId = value(after: "--image-id", in: args, default: "conos-base")
    let instanceId = value(after: "--instance-id", in: args, default: "default")
    let networkMode = value(after: "--network-mode", in: args, default: "provider_default")
    let imagePath = "\(stateRoot)/images/\(imageId)/disk.img"
    return [
        ("schema_version", .string(helperVersion)),
        ("platform", .string("macos")),
        ("provider", .string("managed")),
        ("state_root", .string(stateRoot)),
        ("image_id", .string(imageId)),
        ("instance_id", .string(instanceId)),
        ("network_mode", .string(networkMode)),
        ("virtualization_framework_available", .bool(virtualizationFrameworkAvailable())),
        ("base_image_path", .string(imagePath)),
        ("base_image_present", .bool(fileExists(imagePath))),
        ("no_host_fallback", .bool(true)),
    ]
}

func payloadValue(_ key: String, in payload: [(String, JSONValue)]) -> JSONValue? {
    for (payloadKey, payloadValue) in payload where payloadKey == key {
        return payloadValue
    }
    return nil
}

func payloadString(_ key: String, in payload: [(String, JSONValue)]) -> String {
    if case .string(let value)? = payloadValue(key, in: payload) {
        return value
    }
    return ""
}

func payloadBool(_ key: String, in payload: [(String, JSONValue)]) -> Bool {
    if case .bool(let value)? = payloadValue(key, in: payload) {
        return value
    }
    return false
}

func runReport(args: [String]) -> Int32 {
    var payload = providerPayload(args: args)
    let helperReady = payloadBool("virtualization_framework_available", in: payload)
    payload.append(("status", .string(helperReady ? "READY_FOR_IMAGE" : "UNAVAILABLE")))
    payload.append(("reason", .string(helperReady ? "" : "Apple Virtualization.framework is unavailable")))
    jsonLine(payload)
    return helperReady ? 0 : 78
}

func runInit(args: [String]) -> Int32 {
    var payload = providerPayload(args: args)
    let stateRoot = payloadString("state_root", in: payload)
    let directories = [
        "\(stateRoot)/images",
        "\(stateRoot)/instances",
        "\(stateRoot)/snapshots",
        "\(stateRoot)/overlays",
        "\(stateRoot)/logs",
    ]
    for directory in directories {
        if !ensureDir(directory) {
            eprint("managed VM init failed while creating \(directory)")
            return 74
        }
    }
    payload.append(("status", .string("INITIALIZED")))
    payload.append(("created_directories", .array(directories)))
    jsonLine(payload)
    return 0
}

func runBoot(args: [String]) -> Int32 {
    var payload = providerPayload(args: args)
    let stateRoot = payloadString("state_root", in: payload)
    let instanceId = payloadString("instance_id", in: payload)
    let baseImagePath = payloadString("base_image_path", in: payload)
    let instanceRoot = "\(stateRoot)/instances/\(instanceId)"
    let overlayPath = "\(instanceRoot)/overlay.img"

    guard payloadBool("virtualization_framework_available", in: payload) else {
        payload.append(("status", .string("UNAVAILABLE")))
        payload.append(("reason", .string("Apple Virtualization.framework is unavailable")))
        jsonLine(payload)
        return 78
    }
    guard fileExists(baseImagePath) else {
        payload.append(("status", .string("UNAVAILABLE")))
        payload.append(("reason", .string("managed VM base image is missing")))
        payload.append(("overlay_path", .string(overlayPath)))
        payload.append(("overlay_present", .bool(fileExists(overlayPath))))
        jsonLine(payload)
        return 78
    }
    for directory in [instanceRoot, "\(instanceRoot)/logs", "\(instanceRoot)/workspace", "\(instanceRoot)/snapshots"] {
        if !ensureDir(directory) {
            payload.append(("status", .string("FAILED")))
            payload.append(("reason", .string("failed to create instance directory \(directory)")))
            jsonLine(payload)
            return 74
        }
    }
    let overlayWasPresent = fileExists(overlayPath)
    if !createSparseFile(overlayPath, sizeBytes: 64 * 1024 * 1024) {
        payload.append(("status", .string("FAILED")))
        payload.append(("reason", .string("failed to create overlay artifact")))
        payload.append(("overlay_path", .string(overlayPath)))
        payload.append(("overlay_present", .bool(fileExists(overlayPath))))
        jsonLine(payload)
        return 74
    }
    payload.append(("status", .string("BOOT_CONTRACT_READY_EXEC_UNAVAILABLE")))
    payload.append(("reason", .string("managed VM helper v0.1 prepared image and overlay, but guest agent execution is not implemented")))
    payload.append(("overlay_path", .string(overlayPath)))
    payload.append(("overlay_present", .bool(fileExists(overlayPath))))
    payload.append(("overlay_created", .bool(!overlayWasPresent)))
    payload.append(("virtual_machine_started", .bool(false)))
    payload.append(("guest_agent_ready", .bool(false)))
    payload.append(("execution_ready", .bool(false)))
    payload.append(("no_host_fallback", .bool(true)))
    jsonLine(payload)
    return 0
}

func runStart(args: [String]) -> Int32 {
    var payload = providerPayload(args: args)
    let stateRoot = payloadString("state_root", in: payload)
    let instanceId = payloadString("instance_id", in: payload)
    let baseImagePath = payloadString("base_image_path", in: payload)
    let instanceRoot = "\(stateRoot)/instances/\(instanceId)"
    let overlayPath = "\(instanceRoot)/overlay.img"

    guard payloadBool("virtualization_framework_available", in: payload) else {
        payload.append(("status", .string("UNAVAILABLE")))
        payload.append(("lifecycle_state", .string("unavailable")))
        payload.append(("reason", .string("Apple Virtualization.framework is unavailable")))
        jsonLine(payload)
        return 78
    }
    guard fileExists(baseImagePath) else {
        payload.append(("status", .string("UNAVAILABLE")))
        payload.append(("lifecycle_state", .string("unavailable")))
        payload.append(("reason", .string("managed VM base image is missing")))
        payload.append(("overlay_path", .string(overlayPath)))
        payload.append(("overlay_present", .bool(fileExists(overlayPath))))
        jsonLine(payload)
        return 78
    }
    for directory in [instanceRoot, "\(instanceRoot)/logs", "\(instanceRoot)/workspace", "\(instanceRoot)/snapshots"] {
        if !ensureDir(directory) {
            payload.append(("status", .string("FAILED")))
            payload.append(("lifecycle_state", .string("failed")))
            payload.append(("reason", .string("failed to create instance directory \(directory)")))
            jsonLine(payload)
            return 74
        }
    }
    let overlayWasPresent = fileExists(overlayPath)
    if !createSparseFile(overlayPath, sizeBytes: 64 * 1024 * 1024) {
        payload.append(("status", .string("FAILED")))
        payload.append(("lifecycle_state", .string("failed")))
        payload.append(("reason", .string("failed to create overlay artifact")))
        payload.append(("overlay_path", .string(overlayPath)))
        payload.append(("overlay_present", .bool(fileExists(overlayPath))))
        jsonLine(payload)
        return 74
    }
    payload.append(("status", .string("START_BLOCKED_GUEST_AGENT_OR_BOOT_IMPL_MISSING")))
    payload.append(("lifecycle_state", .string("start_blocked")))
    payload.append(("reason", .string("managed VM helper v0.1 has no Apple Virtualization process lifecycle or guest agent yet")))
    payload.append(("overlay_path", .string(overlayPath)))
    payload.append(("overlay_present", .bool(fileExists(overlayPath))))
    payload.append(("overlay_created", .bool(!overlayWasPresent)))
    payload.append(("process_pid", .string("")))
    payload.append(("virtual_machine_started", .bool(false)))
    payload.append(("guest_agent_ready", .bool(false)))
    payload.append(("execution_ready", .bool(false)))
    payload.append(("no_host_fallback", .bool(true)))
    jsonLine(payload)
    return 0
}

func runRuntimeStatus(args: [String]) -> Int32 {
    var payload = providerPayload(args: args)
    let stateRoot = payloadString("state_root", in: payload)
    let instanceId = payloadString("instance_id", in: payload)
    let runtimePath = "\(stateRoot)/instances/\(instanceId)/runtime.json"
    payload.append(("status", .string(fileExists(runtimePath) ? "RUNTIME_MANIFEST_PRESENT" : "STOPPED")))
    payload.append(("lifecycle_state", .string(fileExists(runtimePath) ? "unknown_from_helper" : "stopped")))
    payload.append(("runtime_manifest_path", .string(runtimePath)))
    payload.append(("runtime_manifest_present", .bool(fileExists(runtimePath))))
    payload.append(("process_pid", .string("")))
    payload.append(("virtual_machine_started", .bool(false)))
    payload.append(("guest_agent_ready", .bool(false)))
    payload.append(("execution_ready", .bool(false)))
    payload.append(("no_host_fallback", .bool(true)))
    jsonLine(payload)
    return 0
}

func runAgentStatus(args: [String]) -> Int32 {
    var payload = providerPayload(args: args)
    let stateRoot = payloadString("state_root", in: payload)
    let instanceId = payloadString("instance_id", in: payload)
    let runtimePath = "\(stateRoot)/instances/\(instanceId)/runtime.json"
    payload.append(("status", .string("GUEST_AGENT_NOT_READY")))
    payload.append(("runtime_manifest_path", .string(runtimePath)))
    payload.append(("runtime_manifest_present", .bool(fileExists(runtimePath))))
    payload.append(("guest_agent_ready", .bool(false)))
    payload.append(("execution_ready", .bool(false)))
    payload.append(("reason", .string("managed VM helper v0.1 has no guest-agent transport yet")))
    payload.append(("no_host_fallback", .bool(true)))
    jsonLine(payload)
    return 78
}

func runAgentExec(args: [String]) -> Int32 {
    var payload = providerPayload(args: args)
    let command = commandAfterSeparator(args)
    guard !command.isEmpty else {
        payload.append(("status", .string("INVALID_AGENT_EXEC_REQUEST")))
        payload.append(("reason", .string("agent-exec requires command after --")))
        jsonLine(payload)
        return 64
    }
    payload.append(("status", .string("EXEC_BLOCKED_GUEST_AGENT_NOT_READY")))
    payload.append(("reason", .string("managed VM helper v0.1 has no guest-agent transport yet; refusing host fallback")))
    payload.append(("guest_agent_ready", .bool(false)))
    payload.append(("execution_ready", .bool(false)))
    payload.append(("no_host_fallback", .bool(true)))
    jsonLine(payload)
    return 78
}

func runStop(args: [String]) -> Int32 {
    var payload = providerPayload(args: args)
    payload.append(("status", .string("STOPPED")))
    payload.append(("lifecycle_state", .string("stopped")))
    payload.append(("reason", .string("managed VM helper v0.1 has no running VM process to stop")))
    payload.append(("process_pid", .string("")))
    payload.append(("virtual_machine_started", .bool(false)))
    payload.append(("guest_agent_ready", .bool(false)))
    payload.append(("execution_ready", .bool(false)))
    payload.append(("no_host_fallback", .bool(true)))
    jsonLine(payload)
    return 0
}

func runExec(args: [String]) -> Int32 {
    let payload = providerPayload(args: args)
    let command = commandAfterSeparator(args)
    guard payloadBool("virtualization_framework_available", in: payload) else {
        eprint("Apple Virtualization.framework is unavailable; refusing host fallback")
        return 78
    }
    guard payloadBool("base_image_present", in: payload) else {
        eprint("managed VM base image is missing at \(payloadString("base_image_path", in: payload)); refusing host fallback")
        return 78
    }
    guard !command.isEmpty else {
        eprint("managed VM exec requires command after --")
        return 64
    }
    eprint("managed VM guest execution is not implemented in helper v0.1; refusing host fallback")
    return 78
}

let args = Array(CommandLine.arguments.dropFirst())
let command = args.first ?? "report"
let rest = Array(args.dropFirst())

switch command {
case "report", "status":
    exit(runReport(args: rest))
case "init":
    exit(runInit(args: rest))
case "boot":
    exit(runBoot(args: rest))
case "start":
    exit(runStart(args: rest))
case "runtime-status":
    exit(runRuntimeStatus(args: rest))
case "agent-status":
    exit(runAgentStatus(args: rest))
case "agent-exec":
    exit(runAgentExec(args: rest))
case "stop":
    exit(runStop(args: rest))
case "exec":
    exit(runExec(args: rest))
default:
    eprint("unsupported managed VM helper command: \(command)")
    exit(64)
}
