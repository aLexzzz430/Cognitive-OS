#import <Foundation/Foundation.h>
#import <Virtualization/Virtualization.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <sys/select.h>
#include <unistd.h>

static NSString *const RunnerVersion = @"conos.apple_virtualization_runner.macos/v0.1";

@interface Options : NSObject
@property(nonatomic, copy) NSString *command;
@property(nonatomic, copy) NSString *stateRoot;
@property(nonatomic, copy) NSString *imageID;
@property(nonatomic, copy) NSString *instanceID;
@property(nonatomic, copy) NSString *diskPath;
@property(nonatomic, copy) NSString *baseImage;
@property(nonatomic, copy) NSString *cloudInitSeedPath;
@property(nonatomic, copy) NSString *efiVariableStorePath;
@property(nonatomic, copy) NSString *bootMode;
@property(nonatomic, copy) NSString *kernelPath;
@property(nonatomic, copy) NSString *initrdPath;
@property(nonatomic, copy) NSString *kernelCommandLine;
@property(nonatomic) int guestAgentPort;
@property(nonatomic, copy) NSString *runtimeManifest;
@property(nonatomic, copy) NSString *networkMode;
@property(nonatomic, copy) NSString *consoleLogPath;
@property(nonatomic, copy) NSString *sharedDirectoryPath;
@property(nonatomic, copy) NSString *sharedDirectoryTag;
@end

@implementation Options
- (instancetype)init {
    self = [super init];
    if (self) {
        _command = @"";
        _stateRoot = @"";
        _imageID = @"conos-base";
        _instanceID = @"default";
        _diskPath = @"";
        _baseImage = @"";
        _cloudInitSeedPath = @"";
        _efiVariableStorePath = @"";
        _bootMode = @"efi_disk";
        _kernelPath = @"";
        _initrdPath = @"";
        _kernelCommandLine = @"";
        _guestAgentPort = 48080;
        _runtimeManifest = @"";
        _networkMode = @"provider_default";
        _consoleLogPath = @"";
        _sharedDirectoryPath = @"";
        _sharedDirectoryTag = @"conos_host";
    }
    return self;
}
@end

static NSString *NowISO(void) {
    NSISO8601DateFormatter *formatter = [[NSISO8601DateFormatter alloc] init];
    formatter.formatOptions = NSISO8601DateFormatWithInternetDateTime;
    return [formatter stringFromDate:[NSDate date]];
}

static NSData *PrefixDataForFile(NSString *path, NSUInteger length) {
    if (path.length == 0 || ![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        return nil;
    }
    NSFileHandle *handle = [NSFileHandle fileHandleForReadingAtPath:path];
    if (handle == nil) {
        return nil;
    }
    NSData *data = [handle readDataOfLength:length];
    [handle closeFile];
    return data;
}

static NSString *HexStringFromData(NSData *data, NSUInteger maxBytes) {
    if (data.length == 0) {
        return @"";
    }
    const unsigned char *bytes = data.bytes;
    NSUInteger count = MIN(data.length, maxBytes);
    NSMutableString *hex = [NSMutableString stringWithCapacity:count * 2];
    for (NSUInteger i = 0; i < count; i++) {
        [hex appendFormat:@"%02x", bytes[i]];
    }
    return hex;
}

static BOOL PrefixLooksLikeArm64LinuxImage(NSData *data) {
    if (data.length < 0x3c) {
        return NO;
    }
    const unsigned char *bytes = data.bytes;
    return bytes[0x38] == 'A' && bytes[0x39] == 'R' && bytes[0x3a] == 'M' && bytes[0x3b] == 'd';
}

static NSString *FormatHintForPrefixData(NSData *data) {
    NSString *hexPrefix = HexStringFromData(data, 16);
    if (PrefixLooksLikeArm64LinuxImage(data)) {
        return @"linux_arm64_image";
    }
    if ([hexPrefix hasPrefix:@"1f8b"]) {
        return @"gzip";
    }
    if ([hexPrefix hasPrefix:@"070701"] || [hexPrefix hasPrefix:@"303730373031"]) {
        return @"cpio_newc";
    }
    if ([hexPrefix hasPrefix:@"4d5a"]) {
        return @"pe_or_efi_wrapped";
    }
    if ([hexPrefix hasPrefix:@"504b0304"]) {
        return @"zip";
    }
    if ([hexPrefix hasPrefix:@"7f454c46"]) {
        return @"elf";
    }
    return hexPrefix.length > 0 ? @"unknown" : @"missing";
}

static NSString *FormatHintForFile(NSString *path) {
    return FormatHintForPrefixData(PrefixDataForFile(path, 64));
}

static NSDictionary *FileDetails(NSString *path) {
    NSString *cleanPath = path ?: @"";
    BOOL exists = cleanPath.length > 0 && [[NSFileManager defaultManager] fileExistsAtPath:cleanPath];
    NSMutableDictionary *details = [@{
        @"path": cleanPath,
        @"present": @(exists)
    } mutableCopy];
    if (!exists) {
        return details;
    }
    NSError *error = nil;
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:cleanPath error:&error];
    if (attributes != nil) {
        NSNumber *size = attributes[NSFileSize];
        if (size != nil) {
            details[@"byte_size"] = size;
        }
        NSDate *modified = attributes[NSFileModificationDate];
        if (modified != nil) {
            NSISO8601DateFormatter *formatter = [[NSISO8601DateFormatter alloc] init];
            formatter.formatOptions = NSISO8601DateFormatWithInternetDateTime;
            details[@"modified_at"] = [formatter stringFromDate:modified] ?: @"";
        }
    }
    NSData *prefixData = PrefixDataForFile(cleanPath, 64);
    NSString *hexPrefix = HexStringFromData(prefixData, 16);
    details[@"magic_hex_prefix"] = hexPrefix;
    details[@"format_hint"] = FormatHintForPrefixData(prefixData);
    return details;
}

static NSMutableDictionary *ErrorDetails(NSError *error) {
    NSMutableDictionary *details = [NSMutableDictionary dictionary];
    if (error == nil) {
        return details;
    }
    details[@"domain"] = error.domain ?: @"";
    details[@"code"] = @(error.code);
    details[@"localized_description"] = error.localizedDescription ?: @"";
    details[@"localized_failure_reason"] = error.localizedFailureReason ?: @"";
    details[@"localized_recovery_suggestion"] = error.localizedRecoverySuggestion ?: @"";

    NSError *underlying = error.userInfo[NSUnderlyingErrorKey];
    if ([underlying isKindOfClass:[NSError class]]) {
        details[@"underlying_error"] = ErrorDetails(underlying);
    }

    NSMutableDictionary *userInfo = [NSMutableDictionary dictionary];
    for (id key in error.userInfo) {
        if (![key isKindOfClass:[NSString class]]) {
            continue;
        }
        id value = error.userInfo[key];
        if ([value isKindOfClass:[NSError class]]) {
            continue;
        }
        userInfo[(NSString *)key] = [value description] ?: @"";
    }
    details[@"user_info"] = userInfo;
    return details;
}

static NSDictionary *RunnerConfigurationPayload(Options *options) {
    NSString *consoleAttachmentKind = options.consoleLogPath.length > 0 ? @"file_handle_serial_port" : @"none";
    return @{
        @"boot_mode": options.bootMode ?: @"",
        @"kernel_command_line": options.kernelCommandLine ?: @"",
        @"kernel": FileDetails(options.kernelPath ?: @""),
        @"initrd": FileDetails(options.initrdPath ?: @""),
        @"writable_disk": FileDetails(options.diskPath ?: @""),
        @"base_image": FileDetails(options.baseImage ?: @""),
        @"cloud_init_seed": FileDetails(options.cloudInitSeedPath ?: @""),
        @"efi_variable_store": FileDetails(options.efiVariableStorePath ?: @""),
        @"console_log": FileDetails(options.consoleLogPath ?: @""),
        @"console_attachment_kind": consoleAttachmentKind,
        @"serial_ports_configured": @(options.consoleLogPath.length > 0 ? 1 : 0),
        @"storage_device_count": @(options.cloudInitSeedPath.length > 0 ? 2 : 1),
        @"network_device_count": @1,
        @"socket_device_count": @1,
        @"directory_sharing_enabled": @(options.sharedDirectoryPath.length > 0),
        @"shared_directory": FileDetails(options.sharedDirectoryPath ?: @""),
        @"shared_directory_tag": options.sharedDirectoryTag ?: @"",
        @"guest_agent_port": @(options.guestAgentPort)
    };
}

static Options *ParseOptions(int argc, const char *argv[]) {
    Options *options = [[Options alloc] init];
    if (argc > 1) {
        options.command = [NSString stringWithUTF8String:argv[1]];
    }
    for (int i = 2; i < argc; i++) {
        NSString *key = [NSString stringWithUTF8String:argv[i]];
        NSString *next = (i + 1 < argc) ? [NSString stringWithUTF8String:argv[i + 1]] : @"";
        if ([key isEqualToString:@"--state-root"]) {
            options.stateRoot = next;
            i++;
        } else if ([key isEqualToString:@"--image-id"]) {
            options.imageID = next;
            i++;
        } else if ([key isEqualToString:@"--instance-id"]) {
            options.instanceID = next;
            i++;
        } else if ([key isEqualToString:@"--disk-path"]) {
            options.diskPath = next;
            i++;
        } else if ([key isEqualToString:@"--base-image"]) {
            options.baseImage = next;
            i++;
        } else if ([key isEqualToString:@"--cloud-init-seed"]) {
            options.cloudInitSeedPath = next;
            i++;
        } else if ([key isEqualToString:@"--efi-variable-store"]) {
            options.efiVariableStorePath = next;
            i++;
        } else if ([key isEqualToString:@"--boot-mode"]) {
            options.bootMode = next;
            i++;
        } else if ([key isEqualToString:@"--kernel-path"]) {
            options.kernelPath = next;
            i++;
        } else if ([key isEqualToString:@"--initrd-path"]) {
            options.initrdPath = next;
            i++;
        } else if ([key isEqualToString:@"--kernel-command-line"]) {
            options.kernelCommandLine = next;
            i++;
        } else if ([key isEqualToString:@"--guest-agent-port"]) {
            options.guestAgentPort = next.intValue;
            i++;
        } else if ([key isEqualToString:@"--runtime-manifest"]) {
            options.runtimeManifest = next;
            i++;
        } else if ([key isEqualToString:@"--network-mode"]) {
            options.networkMode = next;
            i++;
        } else if ([key isEqualToString:@"--console-log"]) {
            options.consoleLogPath = next;
            i++;
        } else if ([key isEqualToString:@"--shared-dir"]) {
            options.sharedDirectoryPath = next;
            i++;
        } else if ([key isEqualToString:@"--shared-tag"]) {
            options.sharedDirectoryTag = next;
            i++;
        }
    }
    return options;
}

static NSMutableDictionary *RuntimePayload(NSString *status,
                                           NSString *lifecycleState,
                                           NSString *reason,
                                           Options *options,
                                           BOOL virtualMachineStarted,
                                           BOOL guestAgentReady,
                                           BOOL executionReady,
                                           BOOL processAlive) {
    return [@{
        @"schema_version": @"conos.managed_vm_provider/v1",
        @"artifact_type": @"managed_vm_runtime",
        @"status": status ?: @"START_FAILED",
        @"lifecycle_state": lifecycleState ?: @"failed",
        @"reason": reason ?: @"",
        @"state_root": options.stateRoot ?: @"",
        @"image_id": options.imageID ?: @"",
        @"instance_id": options.instanceID ?: @"",
        @"network_mode": options.networkMode ?: @"provider_default",
        @"base_image_path": options.baseImage ?: @"",
        @"boot_mode": options.bootMode ?: @"efi_disk",
        @"kernel_path": options.kernelPath ?: @"",
        @"initrd_path": options.initrdPath ?: @"",
        @"writable_disk_path": options.diskPath ?: @"",
        @"cloud_init_seed_path": options.cloudInitSeedPath ?: @"",
        @"cloud_init_seed_present": @([[NSFileManager defaultManager] fileExistsAtPath:options.cloudInitSeedPath ?: @""]),
        @"cloud_init_seed_read_only": @(options.cloudInitSeedPath.length > 0),
        @"guest_console_log_path": options.consoleLogPath ?: @"",
        @"guest_shared_dir_path": options.sharedDirectoryPath ?: @"",
        @"guest_shared_dir_tag": options.sharedDirectoryTag ?: @"",
        @"efi_variable_store_path": options.efiVariableStorePath ?: @"",
        @"efi_variable_store_present": @([[NSFileManager defaultManager] fileExistsAtPath:options.efiVariableStorePath ?: @""]),
        @"process_pid": [NSString stringWithFormat:@"%d", getpid()],
        @"process_alive": @(processAlive),
        @"virtual_machine_started": @(virtualMachineStarted),
        @"guest_agent_ready": @(guestAgentReady),
        @"execution_ready": @(executionReady),
        @"guest_agent_transport": @"virtio-vsock",
        @"guest_agent_port": @(options.guestAgentPort),
        @"launcher_kind": @"apple_virtualization_runner",
        @"runner_version": RunnerVersion,
        @"runner_configuration": RunnerConfigurationPayload(options),
        @"started_at": NowISO(),
        @"no_host_fallback": @YES
    } mutableCopy];
}

static NSMutableDictionary *RuntimePayloadWithError(NSString *status,
                                                    NSString *lifecycleState,
                                                    NSString *reason,
                                                    Options *options,
                                                    BOOL virtualMachineStarted,
                                                    BOOL guestAgentReady,
                                                    BOOL executionReady,
                                                    BOOL processAlive,
                                                    NSError *error) {
    NSMutableDictionary *payload = RuntimePayload(status,
                                                  lifecycleState,
                                                  reason,
                                                  options,
                                                  virtualMachineStarted,
                                                  guestAgentReady,
                                                  executionReady,
                                                  processAlive);
    if (error != nil) {
        payload[@"runner_error"] = ErrorDetails(error);
    }
    return payload;
}

static void WriteRuntime(NSDictionary *payload, NSString *path) {
    if (path.length == 0) {
        return;
    }
    NSURL *url = [NSURL fileURLWithPath:path];
    NSError *error = nil;
    [[NSFileManager defaultManager] createDirectoryAtURL:[url URLByDeletingLastPathComponent]
                             withIntermediateDirectories:YES
                                              attributes:nil
                                                   error:&error];
    if (error != nil) {
        fprintf(stderr, "failed to create runtime directory: %s\n", error.localizedDescription.UTF8String);
        return;
    }
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload
                                                   options:(NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys)
                                                     error:&error];
    if (data == nil || error != nil) {
        fprintf(stderr, "failed to encode runtime manifest: %s\n", error.localizedDescription.UTF8String);
        return;
    }
    [data writeToURL:url options:NSDataWritingAtomic error:&error];
    if (error != nil) {
        fprintf(stderr, "failed to write runtime manifest: %s\n", error.localizedDescription.UTF8String);
    }
}

static NSString *AgentRequestsDirectory(Options *options) {
    return [[[options.stateRoot stringByAppendingPathComponent:@"instances"]
        stringByAppendingPathComponent:options.instanceID ?: @"default"]
        stringByAppendingPathComponent:@"agent-requests"];
}

static void WriteJSONFile(NSDictionary *payload, NSString *path) {
    if (path.length == 0) {
        return;
    }
    NSURL *url = [NSURL fileURLWithPath:path];
    NSError *error = nil;
    [[NSFileManager defaultManager] createDirectoryAtURL:[url URLByDeletingLastPathComponent]
                             withIntermediateDirectories:YES
                                              attributes:nil
                                                   error:&error];
    if (error != nil) {
        fprintf(stderr, "failed to create JSON directory: %s\n", error.localizedDescription.UTF8String);
        return;
    }
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload
                                                   options:(NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys)
                                                     error:&error];
    if (data == nil || error != nil) {
        fprintf(stderr, "failed to encode JSON file: %s\n", error.localizedDescription.UTF8String);
        return;
    }
    NSString *tmpPath = [path stringByAppendingString:@".tmp"];
    [data writeToFile:tmpPath options:NSDataWritingAtomic error:&error];
    if (error != nil) {
        fprintf(stderr, "failed to write temp JSON file: %s\n", error.localizedDescription.UTF8String);
        return;
    }
    [[NSFileManager defaultManager] replaceItemAtURL:url
                                       withItemAtURL:[NSURL fileURLWithPath:tmpPath]
                                      backupItemName:nil
                                             options:0
                                    resultingItemURL:nil
                                               error:&error];
    if (error != nil) {
        [[NSFileManager defaultManager] moveItemAtPath:tmpPath toPath:path error:&error];
    }
}

static NSData *ReadLineWithTimeout(int fd, NSTimeInterval timeoutSeconds) {
    NSMutableData *line = [[NSMutableData alloc] init];
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:MAX(timeoutSeconds, 1.0)];
    while (true) {
        NSTimeInterval remaining = [deadline timeIntervalSinceNow];
        if (remaining <= 0) {
            return nil;
        }
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(fd, &readfds);
        struct timeval timeout;
        timeout.tv_sec = (time_t)remaining;
        timeout.tv_usec = (suseconds_t)((remaining - floor(remaining)) * 1000000.0);
        int selected = select(fd + 1, &readfds, NULL, NULL, &timeout);
        if (selected < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nil;
        }
        if (selected == 0) {
            return nil;
        }
        char byte = 0;
        ssize_t count = read(fd, &byte, 1);
        if (count <= 0) {
            return nil;
        }
        [line appendBytes:&byte length:1];
        if (byte == '\n') {
            return line;
        }
        if (line.length > 10 * 1024 * 1024) {
            return nil;
        }
    }
}

static NSDictionary *JSONObjectFromData(NSData *data) {
    if (data == nil || data.length == 0) {
        return @{};
    }
    NSError *error = nil;
    id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if ([object isKindOfClass:[NSDictionary class]]) {
        return (NSDictionary *)object;
    }
    return @{};
}

static BOOL WriteJSONLineToFD(int fd, NSDictionary *payload) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload options:0 error:&error];
    if (data == nil || error != nil) {
        return NO;
    }
    NSMutableData *line = [data mutableCopy];
    const char newline = '\n';
    [line appendBytes:&newline length:1];
    const uint8_t *bytes = (const uint8_t *)line.bytes;
    NSUInteger remaining = line.length;
    while (remaining > 0) {
        ssize_t written = write(fd, bytes, remaining);
        if (written <= 0) {
            if (errno == EINTR) {
                continue;
            }
            return NO;
        }
        bytes += written;
        remaining -= (NSUInteger)written;
    }
    return YES;
}

API_AVAILABLE(macos(12.0))
@interface AgentSocketDelegate : NSObject <VZVirtioSocketListenerDelegate>
@property(nonatomic, strong) Options *options;
@property(nonatomic, strong) NSMutableArray<VZVirtioSocketConnection *> *connections;
@end

@implementation AgentSocketDelegate
- (instancetype)init {
    self = [super init];
    if (self) {
        _connections = [[NSMutableArray alloc] init];
    }
    return self;
}

- (void)serveRequestSpoolOnFileDescriptor:(int)fd options:(Options *)options API_AVAILABLE(macos(12.0)) {
    NSString *requestRoot = AgentRequestsDirectory(options);
    [[NSFileManager defaultManager] createDirectoryAtPath:requestRoot
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    while (true) {
        @autoreleasepool {
            NSError *listError = nil;
            NSArray<NSString *> *files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:requestRoot error:&listError];
            if (listError != nil) {
                usleep(200000);
                continue;
            }
            NSArray<NSString *> *sortedFiles = [files sortedArrayUsingSelector:@selector(compare:)];
            for (NSString *name in sortedFiles) {
                if (![name hasSuffix:@".request.json"]) {
                    continue;
                }
                NSString *requestID = [name stringByReplacingOccurrencesOfString:@".request.json" withString:@""];
                NSString *requestPath = [requestRoot stringByAppendingPathComponent:name];
                NSString *resultPath = [requestRoot stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.result.json", requestID]];
                if ([[NSFileManager defaultManager] fileExistsAtPath:resultPath]) {
                    continue;
                }
                NSData *requestData = [NSData dataWithContentsOfFile:requestPath];
                NSDictionary *request = JSONObjectFromData(requestData);
                if (request.count == 0) {
                    continue;
                }
                NSMutableDictionary *outbound = [request mutableCopy];
                outbound[@"event_type"] = @"exec";
                outbound[@"request_id"] = requestID;
                NSTimeInterval commandTimeout = 30.0;
                id timeoutValue = outbound[@"timeout_seconds"];
                if ([timeoutValue respondsToSelector:@selector(doubleValue)]) {
                    commandTimeout = MAX([timeoutValue doubleValue], 1.0);
                }
                if (!WriteJSONLineToFD(fd, outbound)) {
                    WriteJSONFile(@{
                        @"schema_version": @"conos.managed_vm_provider/v1",
                        @"event_type": @"exec_result",
                        @"request_id": requestID,
                        @"status": @"FAILED",
                        @"reason": @"failed to write request to guest-agent connection",
                        @"returncode": @78,
                        @"stdout": @"",
                        @"stderr": @"",
                        @"completed_at": NowISO(),
                        @"transport": @"virtio-vsock",
                        @"no_host_fallback": @YES
                    }, resultPath);
                    return;
                }
                NSData *responseLine = ReadLineWithTimeout(fd, commandTimeout + 1.0);
                NSDictionary *response = JSONObjectFromData(responseLine);
                NSMutableDictionary *result = [response mutableCopy];
                if (result.count == 0) {
                    result = [@{
                        @"event_type": @"exec_result",
                        @"status": @"TIMEOUT",
                        @"reason": @"timed out waiting for guest-agent response",
                        @"returncode": @124,
                        @"stdout": @"",
                        @"stderr": @""
                    } mutableCopy];
                }
                result[@"schema_version"] = @"conos.managed_vm_provider/v1";
                result[@"request_id"] = requestID;
                result[@"completed_at"] = NowISO();
                result[@"transport"] = @"virtio-vsock";
                result[@"no_host_fallback"] = @YES;
                if (result[@"status"] == nil) {
                    result[@"status"] = @"FAILED";
                }
                if (result[@"returncode"] == nil) {
                    result[@"returncode"] = [result[@"status"] isEqual:@"COMPLETED"] ? @0 : @1;
                }
                WriteJSONFile(result, resultPath);
            }
        }
        usleep(200000);
    }
}

- (BOOL)listener:(VZVirtioSocketListener *)listener shouldAcceptNewConnection:(VZVirtioSocketConnection *)connection fromSocketDevice:(VZVirtioSocketDevice *)socketDevice API_AVAILABLE(macos(12.0)) {
    [self.connections addObject:connection];
    int fd = connection.fileDescriptor;
    Options *options = self.options;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        NSMutableData *buffer = [[NSMutableData alloc] init];
        char chunk[4096];
        while (true) {
            ssize_t count = read(fd, chunk, sizeof(chunk));
            if (count <= 0) {
                break;
            }
            [buffer appendBytes:chunk length:(NSUInteger)count];
            NSData *newline = [@"\n" dataUsingEncoding:NSUTF8StringEncoding];
            NSRange range = [buffer rangeOfData:newline options:0 range:NSMakeRange(0, buffer.length)];
            if (range.location == NSNotFound) {
                continue;
            }

            NSData *line = [buffer subdataWithRange:NSMakeRange(0, range.location)];
            NSError *error = nil;
            id object = [NSJSONSerialization JSONObjectWithData:line options:0 error:&error];
            NSMutableDictionary *payload = RuntimePayload(@"STARTED",
                                                          @"started",
                                                          @"guest agent ready",
                                                          options,
                                                          YES,
                                                          YES,
                                                          YES,
                                                          YES);
            payload[@"guest_agent_last_seen_at"] = NowISO();
            payload[@"guest_agent_source_port"] = @(connection.sourcePort);
            payload[@"guest_agent_destination_port"] = @(connection.destinationPort);
            if ([object isKindOfClass:[NSDictionary class]]) {
                NSDictionary *message = (NSDictionary *)object;
                id protocolVersion = message[@"protocol_version"];
                id capabilities = message[@"capabilities"];
                id agentVersion = message[@"agent_version"];
                id executionReady = message[@"execution_ready"];
                if (protocolVersion != nil) {
                    payload[@"guest_agent_protocol_version"] = protocolVersion;
                }
                if (agentVersion != nil) {
                    payload[@"guest_agent_version"] = agentVersion;
                }
                if (capabilities != nil) {
                    payload[@"guest_agent_capabilities"] = capabilities;
                }
                if ([executionReady respondsToSelector:@selector(boolValue)]) {
                    payload[@"execution_ready"] = @([executionReady boolValue]);
                }
                payload[@"guest_agent_hello"] = message;
            } else if (error != nil) {
                payload[@"guest_agent_parse_error"] = error.localizedDescription ?: @"invalid guest agent JSON";
            }
            WriteRuntime(payload, options.runtimeManifest);
            const char *ack = "{\"event_type\":\"host_ack\",\"status\":\"accepted\"}\n";
            write(fd, ack, strlen(ack));
            [self serveRequestSpoolOnFileDescriptor:fd options:options];
            break;
        }
    });
    return YES;
}
@end

API_AVAILABLE(macos(12.0))
@interface RunnerDelegate : NSObject <VZVirtualMachineDelegate>
@property(nonatomic, strong) Options *options;
@property(nonatomic, strong) AgentSocketDelegate *agentSocketDelegate;
@property(nonatomic, strong) VZVirtioSocketListener *agentSocketListener;
@end

@implementation RunnerDelegate
- (void)guestDidStopVirtualMachine:(VZVirtualMachine *)virtualMachine API_AVAILABLE(macos(12.0)) {
    WriteRuntime(RuntimePayload(@"STOPPED", @"stopped", @"guest stopped", self.options, NO, NO, NO, NO),
                 self.options.runtimeManifest);
    exit(0);
}

- (void)virtualMachine:(VZVirtualMachine *)virtualMachine didStopWithError:(NSError *)error API_AVAILABLE(macos(12.0)) {
    WriteRuntime(RuntimePayloadWithError(@"STOPPED",
                                         @"stopped",
                                         error.localizedDescription ?: @"guest stopped with error",
                                         self.options,
                                         NO,
                                         NO,
                                         NO,
                                         NO,
                                         error),
                 self.options.runtimeManifest);
    exit(1);
}
@end

API_AVAILABLE(macos(12.0))
static void InstallGuestAgentListener(VZVirtualMachine *virtualMachine, RunnerDelegate *delegate) {
    if (virtualMachine.socketDevices.count == 0) {
        NSMutableDictionary *payload = RuntimePayload(@"STARTED",
                                                      @"started",
                                                      @"VM started but no virtio socket device is available",
                                                      delegate.options,
                                                      YES,
                                                      NO,
                                                      NO,
                                                      YES);
        payload[@"guest_agent_listener_ready"] = @NO;
        WriteRuntime(payload, delegate.options.runtimeManifest);
        return;
    }
    VZSocketDevice *socketDevice = virtualMachine.socketDevices[0];
    if (![socketDevice isKindOfClass:[VZVirtioSocketDevice class]]) {
        NSMutableDictionary *payload = RuntimePayload(@"STARTED",
                                                      @"started",
                                                      @"VM started but socket device is not virtio-vsock",
                                                      delegate.options,
                                                      YES,
                                                      NO,
                                                      NO,
                                                      YES);
        payload[@"guest_agent_listener_ready"] = @NO;
        WriteRuntime(payload, delegate.options.runtimeManifest);
        return;
    }
    VZVirtioSocketListener *listener = [[VZVirtioSocketListener alloc] init];
    AgentSocketDelegate *agentDelegate = [[AgentSocketDelegate alloc] init];
    agentDelegate.options = delegate.options;
    listener.delegate = agentDelegate;
    delegate.agentSocketDelegate = agentDelegate;
    delegate.agentSocketListener = listener;
    [(VZVirtioSocketDevice *)socketDevice setSocketListener:listener forPort:(uint32_t)delegate.options.guestAgentPort];
}

API_AVAILABLE(macos(12.0))
static VZVirtualMachine *MakeVirtualMachine(Options *options, NSError **error) {
    NSURL *diskURL = [NSURL fileURLWithPath:options.diskPath];
    VZDiskImageStorageDeviceAttachment *attachment = [[VZDiskImageStorageDeviceAttachment alloc] initWithURL:diskURL
                                                                                                    readOnly:NO
                                                                                                       error:error];
    if (attachment == nil) {
        return nil;
    }

    VZVirtualMachineConfiguration *configuration = [[VZVirtualMachineConfiguration alloc] init];
    VZGenericPlatformConfiguration *platform = [[VZGenericPlatformConfiguration alloc] init];
    if (@available(macOS 13.0, *)) {
        platform.machineIdentifier = [[VZGenericMachineIdentifier alloc] init];
    }
    configuration.platform = platform;
    NSUInteger maxCPU = VZVirtualMachineConfiguration.maximumAllowedCPUCount;
    configuration.CPUCount = MAX((NSUInteger)1, MIN((NSUInteger)2, maxCPU));
    uint64_t twoGB = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    configuration.memorySize = MAX(VZVirtualMachineConfiguration.minimumAllowedMemorySize,
                                   MIN(twoGB, VZVirtualMachineConfiguration.maximumAllowedMemorySize));
    if ([options.bootMode isEqualToString:@"linux_direct"] || options.kernelPath.length > 0) {
        if (options.kernelPath.length == 0) {
            if (error != NULL) {
                *error = [NSError errorWithDomain:@"ConOSVirtualizationRunner"
                                             code:65
                                         userInfo:@{NSLocalizedDescriptionKey: @"linux_direct boot requires --kernel-path"}];
            }
            return nil;
        }
        NSURL *kernelURL = [NSURL fileURLWithPath:options.kernelPath];
        NSString *kernelFormatHint = FormatHintForFile(options.kernelPath);
        if ([kernelFormatHint isEqualToString:@"pe_or_efi_wrapped"]) {
            if (error != NULL) {
                *error = [NSError errorWithDomain:@"ConOSVirtualizationRunner"
                                             code:67
                                         userInfo:@{
                                             NSLocalizedDescriptionKey: @"linux_direct kernel artifact is EFI/PE-wrapped, not a raw Linux kernel Image",
                                             NSLocalizedFailureReasonErrorKey: @"VZLinuxBootLoader requires a directly bootable Linux kernel artifact for this boot path.",
                                             @"kernel_path": options.kernelPath ?: @"",
                                             @"kernel_format_hint": kernelFormatHint
                                         }];
            }
            return nil;
        }
        VZLinuxBootLoader *linuxBootLoader = [[VZLinuxBootLoader alloc] initWithKernelURL:kernelURL];
        NSString *commandLine = options.kernelCommandLine.length > 0 ? options.kernelCommandLine : [NSString stringWithFormat:@"console=hvc0 root=/dev/vda rw conos.agent=vsock:%d", options.guestAgentPort];
        linuxBootLoader.commandLine = commandLine;
        if (options.initrdPath.length > 0) {
            linuxBootLoader.initialRamdiskURL = [NSURL fileURLWithPath:options.initrdPath];
        }
        configuration.bootLoader = linuxBootLoader;
    } else if (@available(macOS 13.0, *)) {
        VZEFIBootLoader *bootLoader = [[VZEFIBootLoader alloc] init];
        NSString *storePath = options.efiVariableStorePath.length > 0 ? options.efiVariableStorePath : [options.diskPath.stringByDeletingLastPathComponent stringByAppendingPathComponent:@"efi-variable-store.bin"];
        NSURL *storeURL = [NSURL fileURLWithPath:storePath];
        VZEFIVariableStore *variableStore = nil;
        if ([[NSFileManager defaultManager] fileExistsAtPath:storePath]) {
            variableStore = [[VZEFIVariableStore alloc] initWithURL:storeURL];
        } else {
            variableStore = [[VZEFIVariableStore alloc] initCreatingVariableStoreAtURL:storeURL
                                                                               options:0
                                                                                 error:error];
            if (variableStore == nil) {
                return nil;
            }
        }
        bootLoader.variableStore = variableStore;
        configuration.bootLoader = bootLoader;
    } else {
        if (error != NULL) {
            *error = [NSError errorWithDomain:@"ConOSVirtualizationRunner"
                                         code:78
                                     userInfo:@{NSLocalizedDescriptionKey: @"EFI disk boot requires macOS 13 or newer"}];
        }
        return nil;
    }
    NSMutableArray *storageDevices = [NSMutableArray arrayWithObject:[[VZVirtioBlockDeviceConfiguration alloc] initWithAttachment:attachment]];
    if (options.cloudInitSeedPath.length > 0) {
        NSURL *seedURL = [NSURL fileURLWithPath:options.cloudInitSeedPath];
        VZDiskImageStorageDeviceAttachment *seedAttachment = [[VZDiskImageStorageDeviceAttachment alloc] initWithURL:seedURL
                                                                                                           readOnly:YES
                                                                                                              error:error];
        if (seedAttachment == nil) {
            return nil;
        }
        [storageDevices addObject:[[VZVirtioBlockDeviceConfiguration alloc] initWithAttachment:seedAttachment]];
    }
    configuration.storageDevices = storageDevices;
    configuration.entropyDevices = @[[[VZVirtioEntropyDeviceConfiguration alloc] init]];
    configuration.memoryBalloonDevices = @[[[VZVirtioTraditionalMemoryBalloonDeviceConfiguration alloc] init]];
    configuration.socketDevices = @[[[VZVirtioSocketDeviceConfiguration alloc] init]];
    if (options.consoleLogPath.length > 0) {
        NSURL *consoleURL = [NSURL fileURLWithPath:options.consoleLogPath];
        [[NSFileManager defaultManager] createDirectoryAtURL:[consoleURL URLByDeletingLastPathComponent]
                                 withIntermediateDirectories:YES
                                                  attributes:nil
                                                       error:nil];
        if (![[NSFileManager defaultManager] fileExistsAtPath:options.consoleLogPath]) {
            [[NSData data] writeToURL:consoleURL options:NSDataWritingAtomic error:nil];
        }
        NSFileHandle *inputHandle = [NSFileHandle fileHandleForReadingAtPath:@"/dev/null"];
        NSFileHandle *outputHandle = [NSFileHandle fileHandleForWritingAtPath:options.consoleLogPath];
        [outputHandle seekToEndOfFile];
        VZFileHandleSerialPortAttachment *consoleAttachment = [[VZFileHandleSerialPortAttachment alloc]
            initWithFileHandleForReading:inputHandle
                    fileHandleForWriting:outputHandle];
        if (consoleAttachment == nil) {
            if (error != NULL) {
                *error = [NSError errorWithDomain:@"ConOSVirtualizationRunner"
                                             code:66
                                         userInfo:@{NSLocalizedDescriptionKey: @"failed to open guest console log attachment"}];
            }
            return nil;
        }
        VZVirtioConsoleDeviceSerialPortConfiguration *console = [[VZVirtioConsoleDeviceSerialPortConfiguration alloc] init];
        console.attachment = consoleAttachment;
        configuration.serialPorts = @[console];
    }
    if (options.sharedDirectoryPath.length > 0) {
        NSString *tag = options.sharedDirectoryTag.length > 0 ? options.sharedDirectoryTag : @"conos_host";
        if (![VZVirtioFileSystemDeviceConfiguration validateTag:tag error:error]) {
            return nil;
        }
        NSURL *shareURL = [NSURL fileURLWithPath:options.sharedDirectoryPath];
        [[NSFileManager defaultManager] createDirectoryAtURL:shareURL
                                 withIntermediateDirectories:YES
                                                  attributes:nil
                                                       error:nil];
        VZSharedDirectory *directory = [[VZSharedDirectory alloc] initWithURL:shareURL readOnly:NO];
        VZSingleDirectoryShare *share = [[VZSingleDirectoryShare alloc] initWithDirectory:directory];
        VZVirtioFileSystemDeviceConfiguration *fileSystem = [[VZVirtioFileSystemDeviceConfiguration alloc] initWithTag:tag];
        fileSystem.share = share;
        configuration.directorySharingDevices = @[fileSystem];
    }

    VZVirtioNetworkDeviceConfiguration *network = [[VZVirtioNetworkDeviceConfiguration alloc] init];
    network.attachment = [[VZNATNetworkDeviceAttachment alloc] init];
    configuration.networkDevices = @[network];

    if (![configuration validateWithError:error]) {
        return nil;
    }
    return [[VZVirtualMachine alloc] initWithConfiguration:configuration];
}

API_AVAILABLE(macos(12.0))
static int RunVirtualMachine(Options *options) {
    if (options.diskPath.length == 0) {
        WriteRuntime(RuntimePayload(@"START_FAILED", @"failed", @"missing --disk-path", options, NO, NO, NO, NO),
                     options.runtimeManifest);
        return 64;
    }
    if (![[NSFileManager defaultManager] fileExistsAtPath:options.diskPath]) {
        WriteRuntime(RuntimePayload(@"START_FAILED", @"failed", @"disk image does not exist", options, NO, NO, NO, NO),
                     options.runtimeManifest);
        return 66;
    }

    NSError *error = nil;
    VZVirtualMachine *virtualMachine = MakeVirtualMachine(options, &error);
    if (virtualMachine == nil) {
        WriteRuntime(RuntimePayloadWithError(@"START_FAILED",
                                             @"failed",
                                             error.localizedDescription ?: @"invalid VM configuration",
                                             options,
                                             NO,
                                             NO,
                                             NO,
                                             NO,
                                             error),
                     options.runtimeManifest);
        return 70;
    }

    RunnerDelegate *delegate = [[RunnerDelegate alloc] init];
    delegate.options = options;
    virtualMachine.delegate = delegate;

    [virtualMachine startWithCompletionHandler:^(NSError * _Nullable errorOrNil) {
        if (errorOrNil != nil) {
            WriteRuntime(RuntimePayloadWithError(@"START_FAILED",
                                                 @"failed",
                                                 errorOrNil.localizedDescription ?: @"Virtualization start failed",
                                                 options,
                                                 NO,
                                                 NO,
                                                 NO,
                                                 NO,
                                                 errorOrNil),
                         options.runtimeManifest);
            exit(70);
        }
        NSMutableDictionary *payload = RuntimePayload(@"STARTED",
                                                      @"started",
                                                      @"Apple Virtualization VM start callback succeeded; waiting for guest agent",
                                                      options,
                                                      YES,
                                                      NO,
                                                      NO,
                                                      YES);
        payload[@"guest_agent_listener_ready"] = @YES;
        WriteRuntime(payload, options.runtimeManifest);
        InstallGuestAgentListener(virtualMachine, delegate);
    }];

    [[NSRunLoop mainRunLoop] run];
    return 0;
}

static void PrintReport(void) {
    NSDictionary *payload = @{
        @"schema_version": @"conos.managed_vm_provider/v1",
        @"operation": @"runner_report",
        @"status": @"AVAILABLE",
        @"runner_version": RunnerVersion,
        @"supported_commands": @[@"run"],
        @"no_host_fallback": @YES
    };
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload
                                                   options:(NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys)
                                                     error:nil];
    if (data != nil) {
        [[NSFileHandle fileHandleWithStandardOutput] writeData:data];
        [[NSFileHandle fileHandleWithStandardOutput] writeData:[@"\n" dataUsingEncoding:NSUTF8StringEncoding]];
    }
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        Options *options = ParseOptions(argc, argv);
        if ([options.command isEqualToString:@"run"]) {
            if (@available(macOS 12.0, *)) {
                return RunVirtualMachine(options);
            }
            WriteRuntime(RuntimePayload(@"START_FAILED",
                                        @"failed",
                                        @"Apple Virtualization requires macOS 12 or newer",
                                        options,
                                        NO,
                                        NO,
                                        NO,
                                        NO),
                         options.runtimeManifest);
            return 78;
        }
        PrintReport();
        return 0;
    }
}
