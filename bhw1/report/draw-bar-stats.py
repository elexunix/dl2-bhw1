# Me:

#elexunixx@black-hp:~/astudy/hse/year4/generic/dl2/bhw1/report$ wc -l scores-*
# 100 scores-chatgpt-consistency
# 100 scores-chatgpt-fluency
# 100 scores-gpt2xl-consistency
# 100 scores-gpt2xl-fluency
# 100 scores-my-consistency
# 100 scores-my-fluency
# 600 total
#elexunix@black-hp:~/astudy/hse/year4/generic/dl2/bhw1/report$ head scores-my-consistency 
#4
#7
#2
#5
#7
#6
#8
#6
#4
#7
#elexunix@black-hp:~/astudy/hse/year4/generic/dl2/bhw1/report$ head scores-gpt2xl-fluency 
#7
#6
#7
#7
#7
#6
#5
#7
#5
#6
#You see the structure of my data (3 x 2 x 100 numbers in 6 files), write python code that plots bar histogram of each 100 scores, 2 rows 3 columns of plots, label everything there

# GPT4:

import matplotlib.pyplot as plt
import numpy as np

# Define the file names
file_names = [
  "scores-my-consistency",
  "scores-my-fluency",
  "scores-gpt2xl-consistency",
  "scores-gpt2xl-fluency",
  "scores-chatgpt-consistency",
  "scores-chatgpt-fluency",
]

# Create a figure with 2 rows and 3 columns
fig, axs = plt.subplots(2, 3, figsize=(15,10))

# Iterate over the file names and create a histogram for each
for i, file_name in enumerate(file_names):
  # Load the data from the file
  data = np.loadtxt(file_name)

  # Calculate which subplot to use
  row = i % 2
  col = i // 2

  # Plot the histogram
  axs[row, col].hist(data, bins=range(1, 11), edgecolor='black')

  # Set the title of the subplot
  axs[row, col].set_title(file_name)
  print(f'Average {file_name} score: {np.mean(data)}')

# Add a title for the whole figure
fig.suptitle('Scores Histograms')

# Add labels for x and y axis
for ax in axs.flat:
  ax.set(xlabel='Score', ylabel='Frequency')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
  ax.label_outer()

# Save the plot
plt.savefig('bar-hists.png')
