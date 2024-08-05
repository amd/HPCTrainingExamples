# Parameters that are used throughout the various scripts
## Peak theoretical bandwidth of a MI250 GCD
peak = 1638.4


class Metric:
    def __init__(self, t, sum, min, max, mean, unit):
        self.type = t
        self.sum = sum
        self.min = min
        self.max = max
        self.mean = mean
        self.unit = unit
        pass

    def __str__(self):
        return f"min = {self.min} max={self.max} mean={self.mean} sum={self.sum} {self.unit}"


def get_metrics(d):
    out = {}
    for i, field in enumerate(d["field"]):
        out[field] = Metric(d["type"][i], d["sum"][i], d["min"][i], d["max"][i], d["mean"][i],
        d["unit"][i])

    return out

def filename(study, nx, ny, nz, name, r, align, nw, offset):
    return f"{study}/{nx}x{ny}x{nz}/{name}_{r}_align_{align}_nw_{nw}_use_offset_{offset}/results.toml"

def read_kernel_data(filename, kernel):
    import tomli
    with open(filename, "rb") as t:
        tdict = tomli.load(t)
    metrics = get_metrics(tdict[kernel])
    return metrics

def read_kernel_seq(filenames, kernel, metric):
    out = []
    for fn in filenames:
        met = read_kernel_data(fn, kernel)
        out.append(met[metric].mean)
    return out

def gcells(dur, size):
    return (size / 1e9) / (dur / 1e3)

def format(*args):
    vals = []
    for a in args:
        if str(a) == a:
            vals.append(a)
        # Display as int
        elif a == int(a):
            vals.append("%d" % a)
        # Display as float
        else:
            vals.append(f"%2.2f" % a)
    return vals

def print_csv(*args):
    vals = format(*args)
    print(",".join(vals))

def print_markdown(*args):
    vals = format(*args)
    print("|" + "|".join(vals) + "|")

def print_markdown_header(*args):
    vals = format(*args)
    head = ["---" for a in args]
    print("|" + "|".join(vals) + "|")
    print("|" + "|".join(head) + "|")

def bwpop(bw, peak=1638.4):
    #Bandwidth Percentage of Peak (PoP)
    return bw / peak * 100.0


