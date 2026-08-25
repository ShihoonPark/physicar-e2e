# High-Speed Temporal PilotNet V9 V1

V9 is a clean structural A/B experiment using only preserved High-Speed V8 data sources. It converts each trajectory independently into causal `[t-2, t-1, t]` sequences with a frozen 120 ms maximum adjacent gap. The target is Expert steering at `t`; no boundary crossing, history padding, new bags, or further DAgger is permitted.

Each frame receives the canonical RGB crop/resize and RGB-to-YUV normalization independently, then the three CHW tensors are concatenated oldest-to-current into `9×66×200`. The established PilotNet changes only its first convolution from 3 to 9 input channels, yielding exactly 255,819 parameters. V9 trains from scratch with the V8 training sources and unchanged optimization semantics.
