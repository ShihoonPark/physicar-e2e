# Temporal PilotNet V9 training

PASS. V9 changes only conv1 input channels from 3 to 9; all later layers are unchanged. Parameter count is exactly 255,819 (+3,600). Input is `N×9×66×200`, ordered oldest-to-current after independent canonical RGB→YUV preprocessing. Training started from scratch with MSE, Adam, 1e-3, batch 64, seed 20260824 and the unchanged early-stopping rule. Best epoch was 17; training stopped after 24 epochs.

Matched metrics are `MAE / RMSE / bias / max / correlation / corrective ratio`:

| Stratum (count) | V8 current frame | V9 causal temporal |
|---|---|---|
| Nominal validation (482) | .01129/.01836/.00380/.09614/.99328/1.0315 | .01063/.01784/.00108/.09578/.99329/.9898 |
| Nominal holdout (481) | .01202/.01955/.00434/.09793/.99249/1.0407 | .01073/.01747/.00140/.07622/.99349/.9990 |
| DAgger1-B (41) | .02688/.04065/.01078/.12287/.98363/.9588 | .02130/.03271/.00127/.11467/.98865/.9955 |
| DAgger2-B (44) | .04047/.06407/-.00724/.22550/.96847/.9453 | .03533/.05200/-.01165/.16576/.98021/.9573 |
| DAgger3-B (1,011) | .02080/.04388/.00447/.46254/.97315/.9985 | .01817/.04060/.00225/.48289/.97680/.9799 |

DAgger3-B V8→V9 MAE: 85–90% `.02048→.01801` (990), 90–95% `.03835→.02067` (7), 95–100% `.03475→.02837` (14). The optional `[t,t,t]` V9 ablation had MAE .02086 versus causal V9 .01817, indicating that temporal differences contributed rather than only the larger conv1.

Checkpoint: 1,032,805 bytes, SHA-256 `1cded5fcc7f3d13242de096c4868fc576d03fa6bc86df6e5c8c7c235d9faa6cc`. ONNX: 1,026,938 bytes, SHA-256 `7f6aa4c2d8c9b3615c580f065660c674efff94ff4cd0b9bdc9357df904000888`. ONNX checker and `batch×9×66×200 → batch×1` contract passed. PyTorch↔ONNX maximum difference was `6.24e-8 rad`.
