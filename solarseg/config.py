"""Per-class and per-architecture configuration.

Each solar feature is observed in a different SDO channel and has a very different
foreground fraction, so the positive-class weight of the BCE loss is set per class.
"""

CLASSES = {
    # name           SDO channel (file-name token)  pos_weight  Korean
    "coronal_hole": dict(channel="AIA_193", pos_weight=5.0,  ko="코로나홀",
                         desc="AIA 193 Å (EUV) dark coronal-hole regions"),
    "sunspot":      dict(channel="HMI_Ic",  pos_weight=40.0, ko="흑점",
                         desc="HMI continuum intensity dark sunspots"),
    "prominence":   dict(channel="AIA_304", pos_weight=14.0, ko="홍염",
                         desc="AIA 304 Å bright off-limb prominences"),
}

ARCHS = ("unet", "deeplabv3", "segformer")

# Default optimiser settings per architecture (overridable from the CLI).
ARCH_DEFAULTS = {
    "unet":      dict(lr=1e-3, bs=16, epochs=40),
    "deeplabv3": dict(lr=3e-4, bs=16, epochs=40),
    "segformer": dict(lr=1e-3, bs=16, epochs=40),
}

IMAGE_SIZE = 512
