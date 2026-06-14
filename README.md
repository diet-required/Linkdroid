# Linkdroid

# 🤖 Emotion-Driven Desktop Agent (WIP)

> ⚠️ **Disclaimer:** This project is heavily **Work In Progress (WIP)**. My bad, my bad;;;; 
> I'll finish it and push the code ASAP! Stay tuned XD

---

## 📜 License Notice (Important!)

* 🐍 **Python Code:** Distributed under the **MIT License**. Feel free to use it!
* 🧠 **Model Files:** **NOT MIT.** ❌ **NO Commercial Use Allowed.** ❌ (Sorry about that..)
* *More detailed terms and conditions will be announced later. But for now, remember: MIT applies ONLY to the Python code!*

---

## 🤔 So, what on earth is this?

TL;DR: It's a **desktop agent that genuinely feels emotions**. Like, for real.
My ultimate goal is to build a pocket-sized, desktop companion that acts just like a human friend (mostly because I don't have any IRL friends... 🥲).

### ⚙️ Architecture & Tech Stack

To make it fast enough to live on your desktop, I designed a hybrid pipelining structure: 
**Embedding ➡️ Lightweight LLM ➡️ Heavyweight LLM** — and yes, powered by the latest **LangGraph**!

Since LangGraph can be a bit of a heavyweight resource hog, I optimized the emotion processing to run efficiently on your desktop:

1.  **Emotion Processing (Embedding):** Massive LLMs have too much "noise" and overhead. Since emotion matching is basically a similarity search anyway, I figured, *“Why not just use Embeddings?”* It’s way cleaner and faster.
2.  **Behavioral Guidelines & Routing (Lightweight LLM):** If you feed raw emotion vectors or numbers straight to a smaller model, it gets confused and crashes. To fix this, I created a "Behavioral Guideline" mapping for each emotional distribution. The lightweight model simply handles the combination and routing.
3.  **Action & Execution (Heavyweight LLM):** Once the lightweight model passes the refined guidelines, the heavyweight LLM takes over to freely use tools and generate the final, natural response.
4.  **Memory Management:** Even the conversation summary and memory compaction are handled by the lightweight model to save juice.

---

## ⏳ Coming Soon...
I'm baking this as fast as I can. Sit tight and hang tight! 🚀
