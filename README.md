# Linkdroid

# 🤖 Emotion-Driven Desktop Agent (WIP)

<p align="center">
  <a href="#english">English</a> | <a href="#한국어">한국어</a>
</p>

---

<div id="english">

## 🇺🇸 English Version

> ⚠️ **Disclaimer:** This project is heavily **Work In Progress (WIP)**. My bad, my bad;;;; 
> I'll finish it and push the code ASAP! Stay tuned XD

### 📜 License Notice (Important!)
* 🐍 **Python Code:** Distributed under the **MIT License**. Feel free to use it!
* 🧠 **Model Files:** **NOT MIT.** ❌ **NO Commercial Use Allowed.** ❌ (Sorry about that..)
* *More detailed terms will be announced later. Just remember: MIT applies ONLY to the Python code!*

### 🤔 So, what on earth is this?
TL;DR: It's a **desktop agent that genuinely feels emotions**. Like, for real.
My ultimate goal is to build a pocket-sized desktop companion that acts just like a human friend (mostly because I don't have any IRL friends... 🥲).

#### ⚙️ Architecture & Tech Stack
To make it light enough to live on your desktop, I designed a hybrid structure: 
**Embedding ➡️ Lightweight LLM ➡️ Heavyweight LLM** — and yes, powered by the latest **LangGraph**!

1. **Emotion Processing (Embedding):** Massive LLMs have too much "noise" and overhead. Since emotion matching is basically a similarity search anyway, I figured, *“Why not just use Embeddings?”* It’s way cleaner and faster.
2. **Behavioral Guidelines & Routing (Lightweight LLM):** If you feed raw emotion numbers straight to a smaller model, it gets confused and crashes. To fix this, I created a "Behavioral Guideline" mapping for each emotional distribution. The lightweight model simply handles the combination.
3. **Action & Execution (Heavyweight LLM):** Once the lightweight model passes the refined guidelines, the heavyweight LLM takes over to freely use tools and generate the final response.
4. **Memory Management:** Even the conversation summary and memory compaction are handled by the lightweight model to save juice.

#### ⏳ Coming Soon...
I'm baking this as fast as I can. Sit tight and hang tight! 🚀

</div>

---

<div id="한국어">

## 🇰🇷 한국어 버전

> ⚠️ **안내:** 이 프로젝트는 아직 **미완성(WIP)** 상태입니다. ㅈㅅㅈㅅ;;;; 
> 호다닥 만들어서 금방 올릴 테니까 일단 기다려봐 주방! XD

### 📜 라이선스 공지 (중요!)
* 🐍 **파이썬 코드:** **MIT 라이선스**입니다. 자유롭게 사용하세요!
* 🧠 **모델링 파일들:** **MIT 적용 제외.** ❌ **상업적 이용 절대 불가.** ❌ (미안하지만서도.. 안 됩니다..)
* *자세한 내용은 추후에 다시 공지하겠지만, 암튼 MIT 라이선스는 'Python 코드'에만 해당한다는 점 꼭 기억해 주세요!*

### 🤔 자, 이 프로젝트가 뭐냐면요
한 줄 요약: **진짜 감정을 느끼는 데스크톱 에이전트**입니다. 가짜 피드백 말고 진짜 감정이요.
거의 인간 친구처럼 소통할 수 있는 바탕화면 친구를 만드는 게 목표예요 (사실 제가 친구가 없어서 만드는 중... 🥲).

#### ⚙️ 아키텍처 및 기술 스택
안 그래도 무거운 **최신식 랭그래프(LangGraph)**를 쓰면서 가벼운 바탕화면 에이전트로 돌리려다 보니, 다음과 같은 하이브리드 구조를 짜게 됐습니다:
**임베딩 ➡️ 경량 모델 ➡️ 대형 모델**

1. **감정 처리 (임베딩):** 대형 모델은 아는 게 너무 많아서 오히려 처리가 지저분하더라고요. 어차피 감정 매칭도 유사도 검색인데, '걍 임베딩 쓰면 끝 아닌가?' 싶어서 감정 처리는 임베딩으로 직행시켰습니다. 훨씬 깔끔해요.
2. **행동 지침 및 라우팅 (경량 모델):** 감정을 숫자 그대로 경량 모델한테 던지니까 얘가 너무 멍청해서 뻗어버리더라고요. 그래서 각 감정 분포별로 '행동 지침'을 미리 만들었습니다. 경량 모델은 이 지침들을 조합만 하도록 역할을 줄였어요.
3. **실행 및 답변 (대형 모델):** 경량 모델이 정제해서 넘겨준 지침을 바탕으로, 대형 모델이 툴(Tool)을 자유롭게 쓰면서 자연스러운 답변을 뱉어냅니다.
4. **기억 관리:** 심지어 대화 기억을 요약하는 것까지 경량 모델이 처리하게 해서 자원을 최대한 아꼈습니다.

#### ⏳ 조금만 기다려주세요!
열심히 굽고 있습니다. 채널 고정! 🚀

</div>
