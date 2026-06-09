# Veronika Wellness Chatbot QA Runner

This Playwright QA agent opens the live website, runs realistic customer
conversations, captures every customer and assistant turn, and writes JSON and
Markdown issue reports.

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

- `qa-tests/screenshots`: screenshots after every user message and bot reply.
- `qa-tests/screenshots/*-issues.txt`: issue annotations linked to failed turns.
- `qa-tests/reports/latest-report.json`: structured report for automation.
- `qa-tests/reports/latest-report.md`: readable report.

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
