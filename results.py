import numpy as np
import matplotlib.pyplot as plt

SNRs = [0, 3, 6, 9, 12, 15, 18]


# All these are for Rayleigh fading
def RayleighData():
    gram_1 = [
        0.50775214,
        0.65745412,
        0.81056303,
        0.85865369,
        0.85723209,
        0.91981612,
        0.9124495,
    ]
    gram_2 = [
        0.37658901,
        0.53702015,
        0.70294763,
        0.75099604,
        0.79079374,
        0.84634026,
        0.85894836,
    ]
    gram_3 = [
        0.28466804,
        0.44585206,
        0.61749247,
        0.66755772,
        0.75911708,
        0.78573879,
        0.80302084,
    ]
    gram_4 = [
        0.24112381,
        0.40198832,
        0.55739169,
        0.64912953,
        0.69551224,
        0.74280735,
        0.74813184,
    ]
    similarity = [
        0.9086144,
        0.92812039,
        0.95411382,
        0.96613016,
        0.97321588,
        0.9765603,
        0.98044282,
    ]
    return (gram_1, gram_2, gram_3, gram_4), similarity


def AWGNData():
    gram_1 = [
        0.70283349,
        0.88428244,
        0.91760153,
        0.92447218,
        0.92678589,
        0.92759877,
        0.92822869,
    ]
    gram_2 = [
        0.5328391,
        0.79918833,
        0.85149742,
        0.86266082,
        0.86697467,
        0.86893965,
        0.8694445,
    ]
    gram_3 = [
        0.42092586,
        0.72580362,
        0.79386881,
        0.8076261,
        0.8129729,
        0.81539157,
        0.81682885,
    ]
    gram_4 = [
        0.33523422,
        0.66244331,
        0.73937818,
        0.75649889,
        0.76343414,
        0.76622023,
        0.76765574,
    ]
    similarity = [
        0.93813832,
        0.97288062,
        0.97968817,
        0.98122087,
        0.98189976,
        0.98198627,
        0.98203899,
    ]
    return (gram_1, gram_2, gram_3, gram_4), similarity


def plot_data():
    a_gram, a_sim = AWGNData()
    r_gram, r_sim = RayleighData()

    fig = plt.figure(constrained_layout=True, figsize=(15, 24))

    fig_a, fig_r = fig.subfigures(2, 1)

    for name, subfig, gram, sim in zip(
        ("AWGN", "Rayleigh"), (fig_a, fig_r), (a_gram, r_gram), (a_sim, r_sim)
    ):
        axes = subfig.subplots(1, 5)

        subfig.suptitle(name)

        for idx, (data, ax) in enumerate(zip(gram, axes[:4])):
            ax.plot(SNRs, data, "*-")
            ax.grid()
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(f"BLEU {idx + 1}-grams")
            ax.set_ylim(0, 1)
            ax.set_xlim(SNRs[0], SNRs[-1])

        sim_ax = axes[4]
        sim_ax.plot(SNRs, sim, "*-")
        sim_ax.grid()
        sim_ax.set_xlabel("SNR (dB)")
        sim_ax.set_ylabel(f"Sentence similarity")
        sim_ax.set_ylim(0, 1)
        sim_ax.set_xlim(SNRs[0], SNRs[-1])

    plt.show(block=True)


if __name__ == "__main__":
    plot_data()
