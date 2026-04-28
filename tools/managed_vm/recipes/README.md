# Managed VM Artifact Recipes

This directory contains built-in managed VM artifact recipe metadata.

Recipes are not trusted because they are bundled. They are trusted only when
every fetched artifact is pinned by an explicit digest and the managed VM
readiness path verifies a live guest agent before execution is enabled.

The current built-in Debian NoCloud recipe is intentionally blocked. Con OS can
generate and attach the cloud-init NoCloud seed that installs or enables the
guest agent, but the bundled recipe must not become executable until it pins a
concrete source disk artifact and digest.
