# High-Speed PilotNet V5 V1

This gated pipeline keeps PilotNet V4 and the 0.50 m/s dataset immutable. It collects 12 independent laps from High-Speed Expert V1 (1.80 m/s, 0.90 m lookahead), extracts causal camera/steering pairs with the canonical extractor, freezes an 8/2/2 episode split, trains the unchanged 252,219-parameter PilotNet architecture from scratch, exports a distinct V5 ONNX, and conditionally validates it at 1.80 m/s.

Raw bags, images, checkpoints, ONNX, and large plots remain under external simulator userdata. Compact collection, dataset, training, and live evidence remains under `results/`.

The first valid V5 policy failure stops live validation without retry or DAgger. A first pass conditionally permits two more independent laps; three valid passes are required to freeze V5.
