#!/bin/sh
grep Consistency ../100-stories-my | awk '{print $2}' > scores-my-consistency
grep Fluency ../100-stories-my | awk '{print $2}' > scores-my-fluency
grep Consistency ../100-stories-gpt2xl | awk '{print $2}' > scores-gpt2xl-consistency
grep Fluency ../100-stories-gpt2xl | awk '{print $2}' > scores-gpt2xl-fluency
grep Consistency ../100-stories-chatgpt | awk '{print $2}' > scores-chatgpt-consistency
grep Fluency ../100-stories-chatgpt | awk '{print $2}' > scores-chatgpt-fluency
wc -l scores-*
