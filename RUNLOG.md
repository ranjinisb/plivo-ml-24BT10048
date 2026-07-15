# Plivo TTS Voice Cloning - Run Log

## Baseline Run
* **Fitness Design:** Naive 50/50 tensor blending (`blend.py`).
* **Settings:** Evaluated stock voices against target reference directory. 
* **Score:** `af_nova` won with **0.6299**. The naive 50/50 blend scored lower (0.5802).
* **Audio Observations:** Currently af_nova sounds robotic compared to the target speaker. Pacing seems to be fine, can be improved but the pitch is very off. 

---

## Run 1: Structured Random Walk (Row-Broadcasted)
* **Fitness Design:** Cosine similarity over 3 different sentence lengths to prevent overfitting.
* **Settings:** Iterations = 150, Initial Step = 0.04, Annealing = 0.98 multiplier per step.
* **Score:** 0.6387
* **Audio Observations:** Not much change compared to before but the score has improved. 

---

## Run 2: [testing a genetic splice (af_nova + if_sara) combined with momentum-based updates.] 
* **Score:** 0.6473
* **Audio Observations:** While better, the tone if off

## Run 3: [Suddenly, a thought arises in my mind that perhaps I have been incorrect trying to make hybrid voices, thus turning to ridge regression, something i probably should've done earlier.]
* **Score:** 0.5360
* **Audio Observations:** It's completely different than the first two, but still far away from the target voice. The pacing has also changed but it's faster now compared to the target 

## Run 4: [Some improvements made to the regression logic to perform it in chunks]
* **Score:** 0.5198
* **Audio Observations:** The score has fallen off again, but the pacing is much better this time around. Perhaps a gender bias can be included to give more weight to the female voices.

## Run 5: [Gender Bias was included]
* **Score:** 0.5327
* **Audio Observations:** While the score is up a bit from the previous run, it seems that the male tonality is still getting more prominence. I'm going to consider only the top 5 voices that matched during the baseline now.

## Run 6: [Some improvements made to the regression logic to perform it in chunks]
* **Score:** 0.6395
* **Audio Observations:** The score has become much better. The consideration of the top 5 voices was better. The pacing is a bit off still. 
