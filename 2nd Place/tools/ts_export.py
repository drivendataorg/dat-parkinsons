"""Export the 2D member's checkpoints to TorchScript so the container needs no
timm at inference (timm is unpinned in the runtime image). Traces on CPU with
the real input shape; verifies traced vs eager on 64 cached images."""
# PRIZE-REPO PATCH (the only two changes to the original tools/ts_export.py):
#  1. The trace / eager-vs-traced check input is read from outputs/crops_v3_f2.npz
#     (the frame-guarded cache the 2D members were trained on, and the frame
#     inference feeds them as 'aligned_f2') instead of outputs/crops_v3_m2.npz,
#     a cache that is not part of this pipeline. The traced graph and frozen
#     weights do not depend on the VALUES of this sample (EfficientNet has no
#     data-dependent control flow); only the printed check numbers do.
#  2. The final print names the file actually written, outputs/<TAG>_meta.json;
#     the original printed a hard-coded 'outputs/e2dmip_meta.json'.
# Run from the repository root:  python tools/ts_export.py <tag>
import sys, numpy as np, torch
sys.path.insert(0, 'src'); import cnn2d
TAG = sys.argv[1] if len(sys.argv) > 1 else 'e2dmip'
z = np.load('outputs/crops_v3_f2.npz'); X = torch.from_numpy(cnn2d.make_images(z['aligned'][:64], views='mip3'))
dev = torch.device('cpu')
for part in ['fold0', 'fold1', 'fold2', 'fold3', 'fold4', 'full0']:
    ck = torch.load(f'outputs/{TAG}_{part}.pt', map_location='cpu', weights_only=False)
    net = cnn2d.build_model(dict(backbone=ck['backbone'], drop=ck.get('drop', 0.3), hemi=bool(ck.get('hemi', False))), pretrained=False)
    net.load_state_dict(ck['state']); net.eval()
    xb = cnn2d.prep(X[:8], int(ck['res']), dev)
    with torch.no_grad():
        traced = torch.jit.trace(net, xb, check_trace=False)
        traced = torch.jit.freeze(traced.eval())
        a = net(cnn2d.prep(X, int(ck['res']), dev)).squeeze(1).numpy()
        b = traced(cnn2d.prep(X, int(ck['res']), dev)).squeeze(1).numpy()
    torch.jit.save(traced, f'outputs/{TAG}_{part}.ts.pt')
    print(f'{part}: max |eager - traced| logit = {np.abs(a-b).max():.2e}  (res {ck["res"]}, views {ck.get("views")})', flush=True)
# metadata the inference path needs without the checkpoint dict
import json; json.dump(dict(res=int(ck['res']), slabs=[list(s) for s in ck['slabs']], views=ck.get('views', 'slabs'), hemi=bool(ck.get('hemi', False)), backbone=ck['backbone']), open(f'outputs/{TAG}_meta.json', 'w'))
print(f'wrote outputs/{TAG}_meta.json')
