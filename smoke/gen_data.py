import json, sys
# Tiny text-only SFT set in MindSpeed-MM hf format: [{"messages":[...], "images":[]}].
# Each sample is made LONG (~1k tokens) so that with neat_packing@cutoff_len=1024 each
# sample ~fills one packed sequence -> N samples ~= N packed seqs. Need >= global_batch
# (world_size*mbs*grad_accum = 8) packed seqs or the BaseRandomBatchSampler divides by zero.
N = 96
base = [
    ("Explain how a neural network learns from data.",
     "A neural network learns by adjusting the weights of connections between its layered nodes. "
     "Training data is fed forward through the network to produce predictions, a loss function measures "
     "the error against the targets, and backpropagation computes gradients of that loss with respect to "
     "every weight. An optimizer then nudges the weights in the direction that reduces the loss. "),
    ("Describe the water cycle in detail.",
     "The water cycle is the continuous movement of water through the environment. Solar energy evaporates "
     "water from oceans, lakes, and soil into vapor, which rises and cools to condense into clouds. "
     "Precipitation returns water to the surface as rain or snow, where it flows through rivers, infiltrates "
     "into groundwater, and eventually returns to the sea, repeating the cycle indefinitely. "),
    ("Give practical advice for improving focus while studying.",
     "To study with better focus, work in distraction-free blocks of time, keep your phone out of reach, "
     "and use a clear goal for each session. Take short breaks to rest your attention, stay hydrated, and "
     "review material actively by summarizing it in your own words rather than rereading passively. "),
]
out = []
for i in range(N):
    q, a = base[i % len(base)]
    # pad the answer with varied filler so each sample is ~1k tokens and lengths differ a bit
    filler = (f"Point {j}: this is an additional explanatory sentence that adds useful context and detail. "
              for j in range(30 + (i % 12)))
    a_long = a + " ".join(filler) + f" (sample {i})"
    out.append({"messages": [{"role": "user", "content": q},
                             {"role": "assistant", "content": a_long}], "images": []})
with open(sys.argv[1] if len(sys.argv) > 1 else "mini_sft.json", "w") as f:
    json.dump(out, f, ensure_ascii=False)
print(f"wrote {len(out)} samples")
