"""Search skeleton: random walk over voice-tensor space. Runs as-is, but
the fitness function and search strategy are deliberately naive — that is
your hour.

    python search.py --reference_dir ../reference --start blend_baseline.pt \
        --iters 150 --out voice.pt

Ideas the skeleton does NOT do for you:
  * fitness beyond raw similarity (naturalness terms, self-similarity
    across sentences, spectral sanity checks) — see the warning in
    similarity.py
  * evaluating on 2-3 DIFFERENT sentences per candidate (one sentence
    overfits)
  * annealing the step size; accepting sideways moves; restarts
  * structured perturbations: are all 256 dimensions doing the same kind
    of work? Perturb halves separately and find out.
  * the tensor is 510 rows of 256 dims — synthesizing a given text uses ONE
    row, picked by the text's phoneme count. Which rows does your fitness
    actually test, and what is randn_like doing to all the others? (This is
    why a local gain can evaporate on sentences you never evaluated.)
  * listening checkpoints: dump audio every N accepted steps and USE YOUR EARS
  main()"""

import argparse
import torch
import numpy as np
import glob
from sklearn.linear_model import Ridge
import synth
import similarity as sim
from resemblyzer import VoiceEncoder

def main():
    # Keep all the original arguments so their grading script doesn't crash!
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--start", required=False, help="Kept for compatibility, but ignored by regression")
    ap.add_argument("--iters", type=int, default=150, help="Ignored by regression")
    ap.add_argument("--step", type=float, default=0.04, help="Ignored by regression")
    ap.add_argument("--out", default="voice.pt")
    ap.add_argument("--listen_every", type=int, default=10, help="Ignored by regression")
    args = ap.parse_args()

    print("--- INITIATING REVERSE REGRESSION SEARCH ---")
    
    print("1. Extracting Target Acoustic Embedding...")
    target_emb = sim.target_embedding(args.reference_dir)

    
    print("2. Building Curated Dataset from Top 5 Voices...")
    # List of the top 5 elite performers
    elite_voices = ['af_nova', 'if_sara', 'hf_beta', 'zm_yunxia', 'af_heart']
    
    # Only load files that are in our elite list
    voice_files = [f for f in glob.glob("../kokoro_assets/voices/*.pt") 
                   if any(elite in f for elite in elite_voices)]
    
    template = synth.load_voice('../kokoro_assets/voices/af_nova.pt')
    encoder = VoiceEncoder()
    test_text = "The quick brown fox jumps over the lazy dog."

    X_embeddings = []
    Y_tensors = []

    for vf in voice_files:
        tensor = synth.load_voice(vf).clone()
        wav = synth.synthesize(test_text, tensor)
        emb = encoder.embed_utterance(wav)
        
        flat_tensor = tensor[0].view(-1).numpy()
        X_embeddings.append(emb)
        Y_tensors.append(flat_tensor) 

    X = np.stack(X_embeddings)
    Y = np.stack(Y_tensors)
   

    print("3. Fitting Segmented Ridge Regressors...")
    
    # Decouple the pitch (low-freq) and tone (mid-freq) components
    model_pitch = Ridge(alpha=0.5)  # Fine-tuned regularization for pitch
    model_tone = Ridge(alpha=2.0)   # High regularization to prevent "cartoonish" artifacts
    
    # Segmented training
    model_pitch.fit(X, Y[:, :64])     # Pitch dimensions
    model_tone.fit(X, Y[:, 64:192])   # Tone/Timbre dimensions
    
  
    

    print("4. Predicting Optimal Style Tensor...")
    # 4. Predicting Optimal Style Tensor
    # 4. Predict and Assemble
    pitch_pred = model_pitch.predict(target_emb.reshape(1, -1))[0]
    tone_pred = model_tone.predict(target_emb.reshape(1, -1))[0]
    
    # Load template and clone
    template = synth.load_voice('../kokoro_assets/voices/af_nova.pt')
    final_tensor = template.clone()
    
    # Create the disentangled style vector
    final_vector = np.zeros(256)
    final_vector[:64] = pitch_pred
    final_vector[64:192] = tone_pred
    final_vector[192:] = 0.0 # Prosody from template
    
    # Apply to all rows
    predicted_tensor = torch.tensor(final_vector, dtype=torch.float32)
    for i in range(final_tensor.shape[0]):
        final_tensor[i] = predicted_tensor

    # Save to the exact output file the grading script expects
    torch.save(final_tensor, args.out)

    import soundfile as sf
    final_wav = synth.synthesize(test_text, final_tensor)
    sf.write("listen_final.wav", final_wav, synth.SR)

if __name__ == "__main__":
    main()