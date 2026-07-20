import sys, time, torch
import torchvision.models as models
b = int(sys.argv[1]) if len(sys.argv) > 1 else 128
arch = sys.argv[2] if len(sys.argv) > 2 else "resnet50"
dev = torch.device("cuda", 0); torch.cuda.set_device(0)
m = getattr(models, arch)().to(dev)
opt = torch.optim.SGD(m.parameters(), lr=0.1)
crit = torch.nn.CrossEntropyLoss().to(dev)
x = torch.randn(b, 3, 224, 224, device=dev)
y = torch.randint(0, 1000, (b,), device=dev)
for i in range(3):
    t = time.time(); opt.zero_grad()
    loss = crit(m(x), y); loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f"warm {arch} b={b} step{i} {time.time()-t:.1f}s", flush=True)
print("WARM_DONE", flush=True)
