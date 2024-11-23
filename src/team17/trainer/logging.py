import io
import math
import typing

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import scipy.stats
import torch
import torchaudio as ta

import wandb


def plot_mel_spectrogram(
    audio: torch.Tensor, sample_rate: int, resolution: int = 256
) -> PIL.Image.Image:
    audio = audio.mean(1, keepdim=True).float()

    hop_length = 512
    n_fft = 512

    # Calculate the number of frequency bins
    n_freqs = n_fft // 2 + 1

    if audio.shape[-1] < n_fft:
        raise ValueError(
            f"audio input is too short ({audio.shape[-1]} samples). Please provide a longer audio input."
        )

    # Use a regular STFT for resolution
    transform = ta.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    ).to(audio.device)

    specs = transform(audio)

    # Convert amplitude to dB
    specs = ta.functional.amplitude_to_DB(
        specs,
        multiplier=10,
        amin=1e-10,
        db_multiplier=float(torch.log10(torch.max(specs.max(), torch.tensor(1e-10)))),
    )
    specs = specs.squeeze(1).detach().cpu().numpy()

    # Calculate frequency values
    freqs = np.linspace(0, sample_rate / 2, n_freqs)

    spec = specs[0]
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    times = np.arange(spec.shape[1]) * hop_length / sample_rate
    ax.pcolormesh(times, freqs, spec, shading="auto", cmap="viridis")

    ax.set_yscale("symlog", linthresh=1000)
    ax.set_ylim(20, sample_rate / 2)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    # Set specific frequency ticks
    freq_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels([f"{f}" for f in freq_ticks])

    plt.title(
        f"Lower Resolution Spectrogram (FFT: {n_fft}, Hop: {hop_length}, Sample Rate: {sample_rate} Hz)"
    )
    plt.tight_layout()
    fig.canvas.draw()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="jpeg")
    plt.close(fig)
    buffer.seek(0)

    image = PIL.Image.open(buffer)
    height = resolution
    width = int(resolution * (image.width / image.height))

    return image.resize((width, height))


def ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
    ema_values = torch.zeros_like(x)
    ema_values[0] = x[0]
    for i in range(1, len(x)):
        ema_values[i] = alpha * x[i] + (1 - alpha) * ema_values[i - 1]
    return ema_values


def plot_spectrum(
    x: torch.Tensor, sample_rate: int, bins: int = 1024, alpha: float = 0.1
):
    x = x.mean(dim=0, keepdim=False)
    x /= torch.max(torch.abs(x))
    n = len(x)
    fft_x_mag = torch.abs(torch.fft.fft(x))[: n // 2]
    freqs = torch.fft.fftfreq(n, d=1 / sample_rate)[: n // 2]
    psd_db = 10 * torch.log10((fft_x_mag**2) / n)
    psd_db = torch.clip(psd_db, -24, 0)

    freq_indices = np.logspace(0, np.log10(len(freqs) - 1), num=bins, dtype=int)
    psd_db_plot = ema(psd_db[freq_indices], alpha)

    plt.figure(figsize=(12, 6))
    plt.style.use("default")
    plt.plot(freqs[freq_indices].numpy(), psd_db_plot.numpy(), color="blue")
    plt.fill_between(
        freqs[freq_indices].numpy(), psd_db_plot.numpy(), -24, color="blue"
    )
    plt.xscale("log")
    plt.xlim(1, sample_rate // 2)
    plt.ylim(-24, 0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title("Power Spectral Density (PSD)")
    plt.grid(True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="jpeg")
    plt.close()
    buffer.seek(0)

    return PIL.Image.open(buffer)


def plot_histogram(tensor: torch.Tensor, samples: int = 1000) -> PIL.Image.Image:
    tensor = tensor.flatten()
    if tensor.size(0) > samples:
        indices = torch.randperm(tensor.size(0))[:samples]
        data = tensor[indices]
    else:
        data = tensor
    data_np = data.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot histogram with density curve
    ax.hist(data_np, bins=30, color="blue", density=True, alpha=0.7)

    # Add KDE using matplotlib's functionality
    kde = scipy.stats.gaussian_kde(data_np)
    x = np.linspace(-4, 4, 1000)  # Fixed x range from -4 to 4
    ax.plot(x, kde(x), color="darkblue")

    # Set fixed axis limits
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    fig.tight_layout(pad=2.0)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="jpeg")
    plt.close(fig)
    buffer.seek(0)
    return PIL.Image.open(buffer)


def compute_statistics(tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.flatten()

    def compute_moments(tensor: torch.Tensor) -> tuple[float, float, float, float]:
        mean = torch.mean(tensor)
        std = torch.std(tensor, unbiased=False)

        mean, std = torch.mean(tensor).item(), torch.std(tensor).item()
        skewness = (torch.mean(((tensor - mean) / std) ** 3)).item()
        kurtosis = (torch.mean(((tensor - mean) / std) ** 4) - 3).item()

        return mean, std, skewness, kurtosis

    def compute_gaussian_entropy(tensor):
        std = torch.std(tensor, unbiased=False)
        entropy = 0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)) * std**2)
        return entropy.item()

    def compute_percentiles(tensor):
        quantiles = torch.tensor([0.15, 0.25, 0.50, 0.75, 0.95])
        percentiles = torch.quantile(tensor, quantiles)
        return {
            "15th_percentile": percentiles[0].item(),
            "25th_percentile": percentiles[1].item(),
            "50th_percentile": percentiles[2].item(),
            "75th_percentile": percentiles[3].item(),
            "95th_percentile": percentiles[4].item(),
        }

    def compute_iqr(tensor):
        quantiles = torch.tensor([0.25, 0.75])
        percentiles = torch.quantile(tensor, quantiles)
        return (percentiles[1] - percentiles[0]).item()

    mean, std, skewness, kurtosis = compute_moments(tensor)
    max_val, min_val = tensor.max().item(), tensor.min().item()
    gaussian_entropy = compute_gaussian_entropy(tensor)
    percentiles = compute_percentiles(tensor)
    power = torch.mean(tensor.square()).item()
    iqr = compute_iqr(tensor)

    stats = {
        "min": min_val,
        "max": max_val,
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "perplexity": math.exp(gaussian_entropy),
        "power": power,
        "iqr": iqr,
        **percentiles,  # Merge percentiles into the main dictionary
    }

    return stats


def log_wandb_data(
    elements: dict[str, list[torch.Tensor | str]],
    flags: dict[
        str,
        list[
            typing.Literal[
                "audio",
                "spectrum",
                "histogram",
                "statistics",
                "visualization",
                "mel_spectrogram",
                "string",
            ]
        ],
    ],
    add_aggregate: bool = False,
    name: str = "",
    step: int = 0,
    sample_rate: int | None = None,
    max_log_n: int = 32,
) -> None:
    table_values: dict[str, list[wandb.Image | wandb.Audio | str]] = {}
    statistics_to_log: dict[str, float] = {}

    for key, value in elements.items():
        types = flags[key]
        if add_aggregate and all(isinstance(v, torch.Tensor) for v in value):
            tensor_value = typing.cast(list[torch.Tensor], value)
            # Convert tensors to float and stack them for proper averaging
            float_tensors = torch.stack([t.float().cpu() for t in tensor_value])
            avg_tensor = torch.mean(float_tensors, dim=0)
            value = tensor_value + [avg_tensor]

            # If statistics is in types and we're adding aggregate, log individual statistics
            if "statistics" in types:
                stats = compute_statistics(avg_tensor)
                for stat_name, stat_value in stats.items():
                    statistics_to_log[f"statistics/{name}_{key}_{stat_name}"] = (
                        stat_value
                    )

        for log_type in types:
            if log_type == "string":
                table_values[f"{key}_string"] = [str(v) for v in value]
            elif log_type == "audio":
                assert sample_rate is not None
                audio_value = [v for v in value if isinstance(v, torch.Tensor)]
                table_values[f"{key}_audio"] = [
                    wandb.Audio(
                        tensor.float().cpu().transpose(-1, -2), sample_rate=sample_rate
                    )
                    for tensor in audio_value
                    if tensor.ndim == 2
                ]
            elif log_type == "spectrum":
                assert sample_rate is not None
                spectrum_value = [v for v in value if isinstance(v, torch.Tensor)]
                table_values[f"{key}_spectrum"] = [
                    wandb.Image(
                        plot_spectrum(tensor.float().cpu(), sample_rate=sample_rate)
                    )
                    for tensor in spectrum_value
                    if tensor.ndim in (2, 3)
                ]
            elif log_type == "histogram":
                histogram_value = [v for v in value if isinstance(v, torch.Tensor)]
                table_values[f"{key}_histogram"] = [
                    wandb.Image(plot_histogram(tensor.float().cpu()))
                    for tensor in histogram_value
                ]
            elif log_type == "statistics":
                statistics_value = [v for v in value if isinstance(v, torch.Tensor)]
                table_values[f"{key}_statistics"] = [
                    "\n".join(
                        f"{k}: {v:0.6f}"
                        for k, v in compute_statistics(tensor.float().cpu()).items()
                    )
                    for tensor in statistics_value
                ]
            elif log_type == "visualization":
                visualization_value = [v for v in value if isinstance(v, torch.Tensor)]
                table_values[f"{key}_visualization"] = [
                    wandb.Image(tensor.float().cpu().numpy().T)
                    for tensor in visualization_value
                    if tensor.numel() <= 2**17
                ]
            elif log_type == "mel_spectrogram":
                assert sample_rate is not None
                mel_spec_value = [v for v in value if isinstance(v, torch.Tensor)]
                table_values[f"{key}_mel_spectrogram"] = [
                    wandb.Image(
                        plot_mel_spectrogram(
                            tensor.float().cpu().unsqueeze(0), sample_rate=sample_rate
                        )
                    )
                    for tensor in mel_spec_value
                ]
            else:
                raise ValueError(f'Unknown column "{log_type}".')

    data = [list(group) for group in zip(*table_values.values())]
    if len(data) > max_log_n:
        data = data[:max_log_n]
    table = wandb.Table(columns=list(table_values.keys()), data=data)
    wandb.log({f"{name}_step_{step}": table}, step=step)

    if statistics_to_log:
        wandb.log(statistics_to_log, step=step)
