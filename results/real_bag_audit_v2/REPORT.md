# Real PhysiCar Bag Audit V2

Audit execution result: **PASS**. This is a complete-bag data audit, not a training result or a claim of real-robot driving success. V1 remains separate historical evidence of the failed/truncated transfer.

## 1. Complete/footer-valid MCAP proof

| Bag | Bytes | SHA-256 | Opening magic | Closing magic | Strict CRC/footer scan | Metadata counts match | Required streams readable | Result |
|---|---:|---|---|---|---|---|---|---|
| bag_01 | 678405210 | `c98fd035af40fb5e698e94f0a87fe3f8c979f4b473c16807621c70aa0b038f04` | yes | yes | yes | yes | yes | **PASS** |
| bag_02 | 1112101420 | `856bcdebb65db968a549625abb528a80ee863e03540fa458b43c711333826636` | yes | yes | yes | yes | yes | **PASS** |
| bag_03 | 472764359 | `f7b3622c0d35e04fcc31e01aa6041bc5ae46c1ed6fdcb700c30900aa7e34438f` | yes | yes | yes | yes | yes | **PASS** |

Each strict non-seeking scan used CRC validation and reached the parsed footer and closing MCAP magic. Counts from every channel matched `metadata.yaml`; the required camera, steering, and speed messages also decoded without failure.

## 2. Complete readable durations and counts

| Bag | Metadata duration | Readable span | Total metadata/readable | Camera | Steering | Speed |
|---|---:|---:|---:|---:|---:|---:|
| bag_01 | 43.447676 s | 43.447676 s | 3043/3043 | 651/651 @ 15.009 Hz | 653/653 @ 15.007 Hz | 653/653 @ 15.007 Hz |
| bag_02 | 72.767350 s | 72.767350 s | 4989/4989 | 1068/1068 @ 14.953 Hz | 1070/1070 @ 14.697 Hz | 1070/1070 @ 14.697 Hz |
| bag_03 | 30.397140 s | 30.397140 s | 2120/2120 | 454/454 @ 14.921 Hz | 454/454 @ 14.923 Hz | 454/454 @ 14.923 Hz |

## 3. Camera contract, rate, and full-frame validation

| Bag | Frames | Contract | FPS | Gap mean/median/p95/max | Timestamp order failures | Payload/preview failures |
|---|---:|---|---:|---|---:|---:|
| bag_01 | 651 | 480x360 rgb8, step=1440, frame_id=camera | 15.009 | mean 0.067 s, median 0.067 s, p95 0.070 s, max 0.085 s | 0 | 0 |
| bag_02 | 1068 | 480x360 rgb8, step=1440, frame_id=camera | 14.953 | mean 0.067 s, median 0.067 s, p95 0.070 s, max 0.358 s | 0 | 0 |
| bag_03 | 454 | 480x360 rgb8, step=1440, frame_id=camera | 14.921 | mean 0.067 s, median 0.067 s, p95 0.070 s, max 0.252 s | 0 | 0 |

All complete camera messages were audited. No crop was applied; in particular, simulator `y=160:360` was not applied to the real images.

Human-approved Real Camera ROI V1 for future extraction is `x=0:480, y=80:360` (480x280), resized to `200x66` with canonical bilinear resize, then existing RGB-to-YUV preprocessing and causal `[t-2,t-1,t]` input. Horizontal cropping and camera undistortion are disabled.

## 4. Complete steering distributions: recorded and converted

The complete bags store a normalized steering COMMAND on a nominal `[-1,+1]` scale. Per the confirmed real-vehicle contract, the whole stream is converted with `steering_rad = steering_recorded * 0.35`; positive is LEFT, negative is RIGHT, and no selective clipping is performed.

### Recorded normalized values

| Bag | Count/rate | min | p01 | p05 | p25 | mean | median | std | p75 | p95 | p99 | max | neg/near-zero/pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bag_01 | 653/15.007 Hz | -0.813342 | -0.658219 | -0.369320 | -0.032634 | 0.174266 | 0.168138 | 0.331887 | 0.346212 | 0.749216 | 0.828031 | 0.910556 | 168/61/424 |
| bag_02 | 1070/14.697 Hz | -0.934881 | -0.780744 | -0.472270 | -0.014632 | 0.069671 | 0.056475 | 0.322598 | 0.131825 | 0.735899 | 0.855310 | 0.901080 | 250/153/667 |
| bag_03 | 454/14.923 Hz | -0.794695 | -0.701820 | -0.501871 | -0.146002 | 0.013896 | 0.056475 | 0.274169 | 0.185638 | 0.402819 | 0.555814 | 0.715336 | 155/26/273 |

The sign counts use the physical near-zero threshold after conversion; sign itself is unchanged by the positive scale factor.

### Converted physical radians (`recorded * 0.35`)

| Bag | min | p01 | p05 | p25 | mean | median | std | p75 | p95 | p99 | max | Outside +/-0.35 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bag_01 | -0.284670 | -0.230377 | -0.129262 | -0.011422 | 0.060993 | 0.058848 | 0.116161 | 0.121174 | 0.262225 | 0.289811 | 0.318695 | 0 |
| bag_02 | -0.327208 | -0.273260 | -0.165295 | -0.005121 | 0.024385 | 0.019766 | 0.112909 | 0.046139 | 0.257565 | 0.299358 | 0.315378 | 0 |
| bag_03 | -0.278143 | -0.245637 | -0.175655 | -0.051101 | 0.004864 | 0.019766 | 0.095959 | 0.064973 | 0.140987 | 0.194535 | 0.250368 | 0 |

All converted values remain inside the approximate physical +/-0.35 rad range. `/steering` command semantics and sign are confirmed: positive is LEFT and negative is RIGHT.

## 5. Reconciliation of recorded values numerically above +/-0.35

The counts below preserve the original V1 concern by comparing the recorded normalized numbers directly with +/-0.35 before conversion. They are not physical-radian violations after the confirmed x0.35 conversion.

| Bag | Recorded > +/-0.35 / finite | Fraction | Below / above | Converted outside +/-0.35 rad | Temporal episodes | Repeated raw plateaus | Samples with nonzero causal speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| bag_01 | 195/653 | 0.298622 | 35/160 | 0 | 15 | 9 | 195/195 (1.000000) |
| bag_02 | 274/1070 | 0.256075 | 116/158 | 0 | 18 | 22 | 273/274 (0.996350) |
| bag_03 | 92/454 | 0.202643 | 52/40 | 0 | 16 | 2 | 92/92 (1.000000) |

### Temporal locations and speed/driving-phase relation

#### bag_01

At recorded magnitudes above 0.35, causal speed: mean 0.562 m/s, median 0.588 m/s, p95 0.725 m/s, max 0.914 m/s; zero/nonzero=0/195; phases={'active_nonzero_speed': 195}.

| Raw side | Start-end offset | Samples | Recorded min/median/max | Causal speed min/median/max (m/s) | Driving phases |
|---|---:|---:|---|---|---|
| above | +0.340 to +1.884 s | 24 | 0.411930/0.651411/0.904927 | 0.200000/0.503788/0.749803 | {'active_nonzero_speed': 24} |
| above | +2.403 to +3.070 s | 11 | 0.373358/0.558807/0.784415 | 0.524930/0.611014/0.721779 | {'active_nonzero_speed': 11} |
| below | +7.201 to +8.268 s | 17 | -0.576839/-0.507880/-0.368108 | 0.603237/0.633992/0.725425 | {'active_nonzero_speed': 17} |
| above | +11.398 to +11.599 s | 4 | 0.350138/0.354049/0.357943 | 0.711890/0.715294/0.718744 | {'active_nonzero_speed': 4} |
| above | +14.998 to +16.809 s | 28 | 0.378004/0.687649/0.910556 | 0.200000/0.518943/0.730641 | {'active_nonzero_speed': 28} |
| above | +17.660 to +19.060 s | 22 | 0.357943/0.582158/0.700673 | 0.554264/0.600977/0.720852 | {'active_nonzero_speed': 22} |
| above | +21.852 to +24.724 s | 44 | 0.350138/0.746432/0.820688 | 0.200000/0.537897/0.733093 | {'active_nonzero_speed': 44} |
| above | +26.589 to +26.989 s | 7 | 0.362487/0.402666/0.435761 | 0.669369/0.686855/0.724557 | {'active_nonzero_speed': 7} |
| below | +27.789 to +28.526 s | 12 | -0.813342/-0.701583/-0.391391 | 0.515405/0.554118/0.723034 | {'active_nonzero_speed': 12} |
| above | +30.788 to +31.182 s | 7 | 0.365684/0.514386/0.600410 | 0.593336/0.630969/0.847880 | {'active_nonzero_speed': 7} |
| below | +33.779 to +33.853 s | 2 | -0.371139/-0.368854/-0.366570 | 0.706928/0.716288/0.725648 | {'active_nonzero_speed': 2} |
| above | +35.119 to +35.119 s | 1 | 0.383935/0.383935/0.383935 | 0.740599/0.740599/0.740599 | {'active_nonzero_speed': 1} |
| below | +36.513 to +36.718 s | 4 | -0.613501/-0.470344/-0.401583 | 0.587961/0.652695/0.913834 | {'active_nonzero_speed': 4} |
| above | +37.444 to +37.584 s | 3 | 0.354049/0.399077/0.467260 | 0.653480/0.714152/0.898038 | {'active_nonzero_speed': 3} |
| above | +38.184 to +38.717 s | 9 | 0.383344/0.608536/0.653861 | 0.571924/0.589989/0.755066 | {'active_nonzero_speed': 9} |

Repeated consecutive recorded-value plateaus above the unscaled +/-0.35 numeric threshold (6-decimal diagnostic):
- recorded `+0.651411` (converted `+0.227994 rad`): +1.206 to +1.884 s, 11 records, phases={'active_nonzero_speed': 11}.
- recorded `-0.576839` (converted `-0.201894 rad`): +7.868 to +8.006 s, 3 records, phases={'active_nonzero_speed': 3}.
- recorded `+0.692981` (converted `+0.242543 rad`): +16.263 to +16.329 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `+0.559934` (converted `+0.195977 rad`): +16.396 to +16.809 s, 7 records, phases={'active_nonzero_speed': 7}.
- recorded `+0.589284` (converted `+0.206249 rad`): +18.195 to +18.327 s, 3 records, phases={'active_nonzero_speed': 3}.
- recorded `+0.749216` (converted `+0.262226 rad`): +22.926 to +22.992 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `+0.749216` (converted `+0.262226 rad`): +23.124 to +23.253 s, 3 records, phases={'active_nonzero_speed': 3}.
- recorded `+0.749216` (converted `+0.262226 rad`): +23.385 to +23.655 s, 5 records, phases={'active_nonzero_speed': 5}.
- recorded `+0.820688` (converted `+0.287241 rad`): +24.122 to +24.391 s, 5 records, phases={'active_nonzero_speed': 5}.

#### bag_02

At recorded magnitudes above 0.35, causal speed: mean 0.573 m/s, median 0.612 m/s, p95 0.722 m/s, max 1.100 m/s; zero/nonzero=1/273; phases={'active_nonzero_speed': 273, 'zero_speed_inside_active_envelope': 1}.

| Raw side | Start-end offset | Samples | Recorded min/median/max | Causal speed min/median/max (m/s) | Driving phases |
|---|---:|---:|---|---|---|
| above | +4.897 to +5.763 s | 14 | 0.399077/0.805080/0.805080 | 0.200000/0.200000/0.902081 | {'active_nonzero_speed': 14} |
| above | +5.964 to +6.298 s | 6 | 0.377171/0.596223/0.713301 | 0.549662/0.595498/0.719357 | {'active_nonzero_speed': 6} |
| below | +17.086 to +19.892 s | 43 | -0.483865/-0.447176/-0.371112 | 0.645381/0.663528/0.717021 | {'active_nonzero_speed': 43} |
| above | +22.223 to +23.621 s | 22 | 0.357943/0.793465/0.901080 | 0.200000/0.521933/0.721076 | {'active_nonzero_speed': 22} |
| below | +27.151 to +27.822 s | 11 | -0.608173/-0.508139/-0.374903 | 0.590138/0.634116/0.721636 | {'active_nonzero_speed': 11} |
| above | +31.158 to +32.218 s | 17 | 0.354049/0.717298/0.890924 | 0.200000/0.529591/0.728213 | {'active_nonzero_speed': 17} |
| above | +34.148 to +35.219 s | 17 | 0.357943/0.580720/0.686798 | 0.559398/0.601586/0.730641 | {'active_nonzero_speed': 17} |
| below | +38.482 to +39.353 s | 14 | -0.934881/-0.790459/-0.374620 | 0.200000/0.516897/0.651017 | {'active_nonzero_speed': 14} |
| below | +40.880 to +41.212 s | 6 | -0.516649/-0.462544/-0.391391 | 0.629924/0.655977/0.735770 | {'active_nonzero_speed': 6} |
| above | +43.354 to +44.357 s | 16 | 0.350138/0.617274/0.883284 | 0.500000/0.586907/0.844326 | {'active_nonzero_speed': 16} |
| below | +45.418 to +46.411 s | 16 | -0.687329/-0.584838/-0.356488 | 0.559200/0.599888/0.759270 | {'active_nonzero_speed': 16} |
| above | +49.558 to +50.286 s | 12 | 0.365684/0.468959/0.549767 | 0.614982/0.652651/0.725810 | {'active_nonzero_speed': 12} |
| below | +51.413 to +52.278 s | 14 | -0.815144/-0.751916/-0.384502 | 0.200000/0.569236/1.100000 | {'active_nonzero_speed': 14} |
| above | +53.810 to +54.615 s | 13 | 0.413938/0.631556/0.767156 | 0.530755/0.580690/0.723431 | {'active_nonzero_speed': 13} |
| above | +55.078 to +56.351 s | 20 | 0.351149/0.622450/0.825234 | 0.511574/0.584341/0.763159 | {'active_nonzero_speed': 20} |
| below | +58.342 to +59.081 s | 12 | -0.391391/-0.386229/-0.351500 | 0.693008/0.695862/0.728915 | {'active_nonzero_speed': 12} |
| above | +66.663 to +67.529 s | 14 | 0.399077/0.805080/0.805080 | 0.200000/0.200000/0.902081 | {'active_nonzero_speed': 14} |
| above | +67.661 to +68.063 s | 7 | 0.351379/0.555814/0.713301 | 0.000000/0.578674/0.715710 | {'active_nonzero_speed': 6, 'zero_speed_inside_active_envelope': 1} |

Repeated consecutive recorded-value plateaus above the unscaled +/-0.35 numeric threshold (6-decimal diagnostic):
- recorded `+0.735348` (converted `+0.257372 rad`): +5.158 to +5.298 s, 3 records, phases={'active_nonzero_speed': 3}.
- recorded `+0.805080` (converted `+0.281778 rad`): +5.357 to +5.763 s, 7 records, phases={'active_nonzero_speed': 7}.
- recorded `-0.472270` (converted `-0.165295 rad`): +17.426 to +17.624 s, 4 records, phases={'active_nonzero_speed': 4}.
- recorded `-0.475292` (converted `-0.166352 rad`): +17.686 to +17.752 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.469228` (converted `-0.164230 rad`): +17.896 to +18.093 s, 4 records, phases={'active_nonzero_speed': 4}.
- recorded `-0.466166` (converted `-0.163158 rad`): +18.159 to +18.226 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.453425` (converted `-0.158699 rad`): +18.421 to +18.493 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.437657` (converted `-0.153180 rad`): +18.826 to +18.885 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.434444` (converted `-0.152055 rad`): +18.952 to +19.152 s, 4 records, phases={'active_nonzero_speed': 4}.
- recorded `-0.444023` (converted `-0.155408 rad`): +19.562 to +19.621 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `+0.901080` (converted `+0.315378 rad`): +22.486 to +22.963 s, 8 records, phases={'active_nonzero_speed': 8}.
- recorded `-0.837354` (converted `-0.293074 rad`): +38.621 to +38.687 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.934881` (converted `-0.327208 rad`): +38.754 to +38.815 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `+0.350138` (converted `+0.122548 rad`): +44.288 to +44.357 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.604621` (converted `-0.211617 rad`): +45.884 to +45.945 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.781563` (converted `-0.273547 rad`): +51.543 to +51.625 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.780376` (converted `-0.273132 rad`): +51.681 to +51.819 s, 3 records, phases={'active_nonzero_speed': 3}.
- recorded `-0.384502` (converted `-0.134576 rad`): +58.414 to +58.480 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.387956` (converted `-0.135785 rad`): +58.680 to +58.741 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.391391` (converted `-0.136987 rad`): +58.807 to +58.948 s, 3 records, phases={'active_nonzero_speed': 3}.
- recorded `+0.735348` (converted `+0.257372 rad`): +66.929 to +67.063 s, 3 records, phases={'active_nonzero_speed': 3}.
- recorded `+0.805080` (converted `+0.281778 rad`): +67.128 to +67.529 s, 7 records, phases={'active_nonzero_speed': 7}.

#### bag_03

At recorded magnitudes above 0.35, causal speed: mean 0.633 m/s, median 0.666 m/s, p95 0.789 m/s, max 0.940 m/s; zero/nonzero=0/92; phases={'active_nonzero_speed': 92}.

| Raw side | Start-end offset | Samples | Recorded min/median/max | Causal speed min/median/max (m/s) | Driving phases |
|---|---:|---:|---|---|---|
| above | +2.401 to +2.401 s | 1 | 0.357943/0.357943/0.357943 | 0.763159/0.763159/0.763159 | {'active_nonzero_speed': 1} |
| above | +6.402 to +6.942 s | 9 | 0.375370/0.440862/0.508245 | 0.633821/0.666747/0.721494 | {'active_nonzero_speed': 9} |
| above | +10.141 to +10.339 s | 4 | 0.402819/0.578225/0.715336 | 0.200000/0.571085/0.612323 | {'active_nonzero_speed': 4} |
| above | +14.134 to +14.200 s | 2 | 0.361822/0.378570/0.395318 | 0.709651/0.824958/0.940265 | {'active_nonzero_speed': 2} |
| above | +15.537 to +15.664 s | 3 | 0.350138/0.357943/0.369529 | 0.705241/0.716437/0.763159 | {'active_nonzero_speed': 3} |
| above | +16.735 to +16.999 s | 5 | 0.360362/0.393370/0.428510 | 0.673128/0.691921/0.726571 | {'active_nonzero_speed': 5} |
| below | +17.539 to +18.530 s | 16 | -0.794695/-0.686788/-0.374620 | 0.200000/0.540208/0.730325 | {'active_nonzero_speed': 16} |
| below | +19.598 to +19.730 s | 3 | -0.371139/-0.360583/-0.360583 | 0.704326/0.710365/0.720866 | {'active_nonzero_speed': 3} |
| above | +20.796 to +20.995 s | 4 | 0.365684/0.423078/0.474117 | 0.650113/0.675972/0.798052 | {'active_nonzero_speed': 4} |
| below | +21.807 to +23.601 s | 28 | -0.649769/-0.476773/-0.367639 | 0.573514/0.648824/0.725225 | {'active_nonzero_speed': 28} |
| below | +23.735 to +23.801 s | 2 | -0.372678/-0.370158/-0.367639 | 0.703454/0.713188/0.722922 | {'active_nonzero_speed': 2} |
| above | +24.335 to +24.335 s | 1 | 0.402819/0.402819/0.402819 | 0.725810/0.725810/0.725810 | {'active_nonzero_speed': 1} |
| above | +25.001 to +25.061 s | 2 | 0.369529/0.425213/0.480896 | 0.705241/0.753196/0.801150 | {'active_nonzero_speed': 2} |
| below | +25.262 to +25.402 s | 3 | -0.456519/-0.434444/-0.394808 | 0.658818/0.691133/0.823635 | {'active_nonzero_speed': 3} |
| above | +26.200 to +26.200 s | 1 | 0.387747/0.387747/0.387747 | 0.814838/0.814838/0.814838 | {'active_nonzero_speed': 1} |
| above | +26.665 to +27.129 s | 8 | 0.370282/0.503563/0.600410 | 0.593336/0.636355/0.781310 | {'active_nonzero_speed': 8} |

Repeated consecutive recorded-value plateaus above the unscaled +/-0.35 numeric threshold (6-decimal diagnostic):
- recorded `-0.766897` (converted `-0.268414 rad`): +17.605 to +17.663 s, 2 records, phases={'active_nonzero_speed': 2}.
- recorded `-0.686788` (converted `-0.240376 rad`): +17.733 to +17.936 s, 4 records, phases={'active_nonzero_speed': 4}.

No actuator-saturation meaning is inferred from a repeated normalized numeric plateau. Direct CDR Float64 decoding independently matched `mcap_ros2` decoding. The raw values remain preserved, and the confirmed whole-stream scaling—not clipping—reconciles them with the physical range.

## 6. Complete speed distributions (m/s)

| Bag | Count/rate | min | p01 | p05 | p25 | mean | median | std | p75 | p95 | p99 | max | negative/zero/positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bag_01 | 653/15.007 Hz | 0.000000 | 0.000000 | 0.200000 | 0.633992 | 0.734677 | 0.768690 | 0.214639 | 0.880728 | 0.977058 | 0.996523 | 0.998500 | 0/17/636 |
| bag_02 | 1070/14.697 Hz | 0.000000 | 0.200000 | 0.512683 | 0.705115 | 0.814245 | 0.890066 | 0.191106 | 0.949204 | 0.990686 | 0.998500 | 1.100000 | 0/2/1068 |
| bag_03 | 454/14.923 Hz | 0.200000 | 0.200000 | 0.592995 | 0.726788 | 0.805925 | 0.833865 | 0.150843 | 0.918648 | 0.970531 | 0.993551 | 0.998168 | 0/0/454 |

The `/speed` unit is confirmed as meters per second (m/s). Only whether it is a command or an actual feedback/measurement remains unresolved. Exact zero is only a diagnostic stationary indicator.

## 7. Timestamp domain

MCAP `log_time` is the single synchronization domain for camera, steering, and speed. Camera headers are diagnostic only because both Float64 streams lack headers; no unproven mixed-clock transform is used.

| Bag | Camera-header backward steps | Header duplicate timestamps | First/last record-minus-header offset | Offset change |
|---|---:|---:|---|---:|
| bag_01 | 0 | 0 | 58635.026703/58635.025103 s | -0.001600 s |
| bag_02 | 1 | 145 | 58978.089177/59039.851986 s | 61.762809 s |
| bag_03 | 1 | 0 | 58642.674489/58673.367683 s | 30.693194 s |

Camera header audit: bag_02 has 1 backward header step(s); bag_02 repeats 145 header timestamp(s); bag_03 has 1 backward header step(s). This independently rules out using camera header time as a shared raw clock with the headerless scalar topics.

## 8. Causal camera-to-label ages

| Bag | Steering age | Speed age | Missing steering | Missing speed | Complete causal pairs |
|---|---|---|---:|---:|---:|
| bag_01 | mean 60.615 ms, median 60.401 ms, p95 66.726 ms, max 78.874 ms | mean 60.571 ms, median 60.355 ms, p95 66.717 ms, max 78.831 ms | 0 | 0 | 651 |
| bag_02 | mean 62.237 ms, median 61.283 ms, p95 67.745 ms, max 356.701 ms | mean 62.191 ms, median 61.234 ms, p95 67.737 ms, max 356.690 ms | 1 | 1 | 1067 |
| bag_03 | mean 62.622 ms, median 61.518 ms, p95 68.188 ms, max 250.872 ms | mean 62.579 ms, median 61.459 ms, p95 68.153 ms, max 250.863 ms | 1 | 1 | 453 |

The join is causal ZOH: for camera time `t`, each label is the latest same-domain scalar record with `t_scalar <= t`.

## 9. Future-label violations

Total future-label violations across all three bags: **0**.

## 10. Full three-frame temporal readiness

| Bag | Candidates | Strict ordered | Current-label ready | Order violations | Adjacent gap distribution | Oldest-current span | Adjacent gaps >120 ms |
|---|---:|---:|---:|---:|---|---|---:|
| bag_01 | 649 | 649 | 649 | 0 | mean 0.067 s, median 0.067 s, p95 0.070 s, max 0.085 s | mean 0.133 s, median 0.133 s, p95 0.136 s, max 0.152 s | 0 |
| bag_02 | 1066 | 1066 | 1066 | 0 | mean 0.067 s, median 0.067 s, p95 0.070 s, max 0.358 s | mean 0.134 s, median 0.133 s, p95 0.137 s, max 0.423 s | 1 |
| bag_03 | 452 | 452 | 452 | 0 | mean 0.067 s, median 0.067 s, p95 0.070 s, max 0.252 s | mean 0.134 s, median 0.133 s, p95 0.137 s, max 0.319 s | 1 |

The simulator 120 ms gap threshold is reported diagnostically and was not applied as a real-data rejection gate.

- bag_02: 0.357921 s between camera #899 and #900 (current frame +60.273 s).
- bag_03: 0.252107 s between camera #446 and #447 (current frame +29.974 s).


## 11. Cross-bag consistency

complete-bag comparison is valid and the camera contract is consistent across all three bags.

- bag_01: 195 normalized recorded steering samples numerically exceed +/-0.35 before the required x0.350000 whole-stream conversion; 0 converted-radian samples remain outside +/-0.35
- bag_02: 274 normalized recorded steering samples numerically exceed +/-0.35 before the required x0.350000 whole-stream conversion; 0 converted-radian samples remain outside +/-0.35
- bag_03: 92 normalized recorded steering samples numerically exceed +/-0.35 before the required x0.350000 whole-stream conversion; 0 converted-radian samples remain outside +/-0.35
- speed medians differ across complete bags (m/s): bag_01=0.768690, bag_02=0.890066, bag_03=0.833865
- steering medians differ across complete bags: bag_01=+0.058848 rad, bag_02=+0.019766 rad, bag_03=+0.019766 rad
- exact-zero speed record counts in complete bags: bag_01=17, bag_02=2, bag_03=0; complete-bag suffix stationarity is observable
- camera timing gap p95 across complete bags: bag_01=0.069506 s, bag_02=0.069608 s, bag_03=0.070370 s

## 12. Candidate active-driving windows

### bag_01 (3 nonzero-speed run(s); exact-zero prefix/suffix records=0/0)

- +0.000 to +1.884 s (1.884 s, 29 records), speed min/median/max (m/s)=0.200000/0.515376/0.998500.
- +2.341 to +16.809 s (14.469 s, 218 records), speed min/median/max (m/s)=0.200000/0.792746/0.996523.
- +17.596 to +43.448 s (25.852 s, 389 records), speed min/median/max (m/s)=0.200000/0.788937/0.998500.

### bag_02 (3 nonzero-speed run(s); exact-zero prefix/suffix records=0/0)

- +0.032 to +5.763 s (5.731 s, 87 records), speed min/median/max (m/s)=0.200000/0.931495/0.972305.
- +5.897 to +67.529 s (61.632 s, 922 records), speed min/median/max (m/s)=0.200000/0.878392/1.100000.
- +67.661 to +72.767 s (5.106 s, 59 records), speed min/median/max (m/s)=0.549662/0.967597/1.100000.

### bag_03 (1 nonzero-speed run(s); exact-zero prefix/suffix records=0/0)

- +0.009 to +30.365 s (30.357 s, 454 records), speed min/median/max (m/s)=0.200000/0.833865/0.998168.

These are diagnostic candidates based only on exact nonzero speed in m/s; they are not extraction filters, and no command-versus-feedback interpretation is inferred.

## 13. Bounded visual previews

- bag_01: `/home/a/physicar-e2e-artifacts/real_bag_audit_v2/previews/bag_01/bag_01_contact_sheet.png` (11 uncropped selected frames; decode failures=0)
- bag_02: `/home/a/physicar-e2e-artifacts/real_bag_audit_v2/previews/bag_02/bag_02_contact_sheet.png` (11 uncropped selected frames; decode failures=0)
- bag_03: `/home/a/physicar-e2e-artifacts/real_bag_audit_v2/previews/bag_03/bag_03_contact_sheet.png` (11 uncropped selected frames; decode failures=0)

Selections span first/early/middle/late/last frames and time-separated positive-LEFT/negative-RIGHT steering-command extrema. Human ROI review is complete: Real Camera ROI V1 preserves y=80:160 far-track curvature, orange center-line vanishing-point information, and early cone visibility. The audit previews remain uncropped evidence.

## 14. Remaining blockers before real dataset extraction

- freeze and test the user-confirmed whole-stream steering conversion steering_rad = steering_recorded * 0.35 in the future real-data extractor (never clip only the raw exceedances)
- approve a real-data temporal-gap and label-staleness policy for 2 adjacent camera gaps above 120 ms and observed causal label ages up to 356.701 ms
- confirm only whether /speed is a command or an actual feedback/measurement value; its m/s unit is already confirmed

## 15. REAL_DATASET_EXTRACTION decision

**NOT YET JUSTIFIED**: the complete-bag audit and ROI approval are complete, but preprocessing, temporal-policy, and /speed command-versus-feedback blockers remain

No dataset extraction or training was performed by this audit.

## 16. Tests

The audit artifact generator does not run the repository test suite. The exact post-audit test command and result are reported in the task handoff.

## 17. Git status

No commit or push was performed. The exact final worktree status is reported in the task handoff.

## Scope attestation

No training, final dataset extraction, simulator use, Docker use, bag modification, odometry requirement, steering clipping, speed-unit assumption, steering-sign assumption, or real-camera ROI application occurred.
