# Managed VM Artifact Recipes

This directory contains built-in managed VM artifact recipe metadata.

Recipes are not trusted because they are bundled. They are trusted only when
every fetched artifact is pinned by an explicit digest and the managed VM
readiness path verifies a live guest agent before execution is enabled.

The default Debian GenericCloud recipe pins a concrete official Debian 12 arm64
RAW disk artifact by sha512. It is the default because the upstream image
includes cloud-init, Python, and socat, which lets Con OS inject the guest agent
through an instance-specific NoCloud seed without asking the user to configure a
VM. Resolving the recipe may download a large disk image, but it is never trusted
until the digest matches and the managed VM readiness path verifies a live guest
agent before execution is enabled.

The older Debian NoCloud recipe remains available as a non-default diagnostic
fixture. It is useful for testing fallback paths where cloud-init is absent, but
it is not a suitable default execution image.
