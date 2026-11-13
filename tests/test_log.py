from windinet.experiment.log import log_metrics, num


def test_num():
    cases = {
        0.058: "0.058",
        28.768: "28.768",
        114.851: "114.85",
        4145.098: "4145.1",
        2268260.75: "2.2683e+06",
        7656707719168: "7.6567e+12",
        0.000012345: "1.2345e-05",
        0.0: "0.000",
    }
    assert {v: num(v) for v in cases} == cases
    assert num(0.23071, ".4f") == "0.2307"
    assert num(4.5, ".1f") == "4.5"


def test_log_metrics_exploding_mean(capsys):
    fields = {"pressure": (0.058, 0.051), "velocityX": (0.102, 0.094), "velocityY": (114.851, 0.21)}
    metrics = {"vrmse_mean": 28.768, "vrmse_std": 107.032, "vrmse_median_mean": 0.0812, "h1_mean": 4145.098}
    metrics |= {"ssim_mean": 0.911, "spectral_low_mean": 2268260.75, "spectral_mid_mean": 106137.93}
    metrics |= {"spectral_high_mean": 17966.48, "power_low_mean": 7656707719168.0}
    metrics |= {"power_mid_mean": 376254988288.0, "power_high_mean": 23028545536.0}
    for f, (mean, median) in fields.items():
        metrics |= {f"vrmse_{f}": mean, f"vrmse_median_{f}": median}
    log_metrics("val 1000", metrics, 0.2355, 840)
    out = capsys.readouterr().out
    assert "(median 0.081)" in out and "| median     p 0.051  u 0.094  v 0.210" in out
    assert max(len(w) for w in out.split()) <= 11
