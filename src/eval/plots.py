#this file is for all the plots
import matplotlib.pyplot as plt

def plot_bleu_scores(SNR, bleu_scores_1gram, bleu_scores_2gram, title="BLEU Score vs. SNR"):
    plt.figure(figsize=(8, 6))
    plt.plot(SNR, bleu_scores_1gram, marker='o', linestyle='-', label='1-gram BLEU Score')
    plt.plot(SNR, bleu_scores_2gram, marker='s', linestyle='--', label='2-gram BLEU Score')
    
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLEU Score")
    plt.legend()
    plt.title(title)
    plt.grid(True)

    plt.savefig('/log/bleu_score_plot.png')

    plt.show()
#####can be updated later for multiple lines
