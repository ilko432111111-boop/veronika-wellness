# Veronika Wellness Chatbot QA Runner

This Playwright QA agent opens the live website, runs realistic customer
conversations, records every customer and assistant turn, and writes
transcript-first QA reports.

## Setup

```powershell
npm install
npm run qa:install
```

## Run

```powershell
npm run qa:chatbot
```

Useful options:

```powershell
npm run qa:chatbot:headed
npm run qa:chatbot:debug
$env:QA_SCENARIO="short-replies"; npm run qa:chatbot
$env:QA_BASE_URL="https://veronika-wellness.onrender.com"; npm run qa:chatbot
$env:QA_TIMEOUT="120000"; npm run qa:chatbot
```

Generated output:

- `qa-tests/reports/latest-transcripts.txt`: plain text transcripts designed for quick review.
- `qa-tests/reports/latest-transcripts.md`: Markdown transcripts that are easy to paste into ChatGPT.
- `qa-tests/reports/latest-report.json`: structured report for automation.

Each transcript includes the scenario name, pass/fail status, detected issues,
the full user/bot conversation, and a scenario summary. The terminal also
prints the pass/fail totals and the paths to all three reports.

Each detected issue includes:

- Confidence: `definite failure`, `likely failure`, or `informational warning`.
- Expected behavior.
- The exact bot reply.
- The reason the detector raised the issue.

`Definite failure` and `likely failure` findings fail a scenario.
`Informational warning` findings are reported without failing the scenario.

## Optional Screenshots

Screenshots are disabled by default so the main output stays lightweight and
text-focused. To capture screenshots after every user message and bot reply,
change the setting near the top of `qa-tests/run-chatbot-qa.js`:

```js
const TAKE_SCREENSHOTS = true;
```

When enabled, screenshots are saved in `qa-tests/screenshots`. Issue annotation
files are written beside the relevant failed-turn screenshots. When disabled,
all tests, issue detection, transcripts, and JSON reporting still run normally.

## Add A Scenario

Edit `qa-tests/scenarios.js` and append a data-only scenario:

```js
{
  name: "customer-asks-location-during-booking",
  messages: [
    { text: "I'd like a relaxing massage." },
    {
      text: "Where are you based?",
      expectAnswerAny: ["Leeds", "Albion Place"]
    }
  ]
}
```

The shared rule engine automatically checks premature handoff wording,
unverified availability claims, repeated questions, autonomous confirmation,
empty replies, meaningless `Thanks.`, and multiple workflow questions.
