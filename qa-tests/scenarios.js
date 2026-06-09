/**
 * Conversation scenarios are intentionally data-only.
 *
 * To add a scenario, append an object with a unique name and a messages array.
 * Optional per-message expectations are evaluated against the bot reply that
 * follows that message.
 */
module.exports = [
  {
    name: "simple-complete-massage-booking",
    messages: [
      { text: "Hi, I'd like to book a relaxing massage." },
      { text: "One hour please." },
      { text: "Tomorrow at 2pm." },
      { text: "My name is Ilko and my phone number is 07700900111." }
    ]
  },
  {
    name: "massage-today-time-later",
    messages: [
      { text: "I want a relaxing massage today." },
      {
        text: "2pm.",
        forbidQuestionTypes: ["date"]
      },
      { text: "One hour." },
      { text: "Ilko, 07700900112." }
    ]
  },
  {
    name: "name-and-phone-given-early",
    messages: [
      { text: "I'm Ilko, 07700900113, and I want to book." },
      {
        text: "A relaxing massage.",
        forbidQuestionTypes: ["name", "phone"]
      },
      {
        text: "One hour tomorrow at 3pm.",
        forbidQuestionTypes: ["name", "phone"]
      }
    ]
  },
  {
    name: "service-list-then-choice",
    messages: [
      {
        text: "What treatments do you do?",
        expectAnswerAny: ["massage", "ems", "ultrasound", "facial", "filler"]
      },
      {
        text: "I'd like a relaxing massage.",
        expectQuestionTypes: ["duration"]
      },
      { text: "One hour tomorrow at 11am." }
    ]
  },
  {
    name: "change-massage-to-lip-filler",
    messages: [
      { text: "I'd like a relaxing massage tomorrow at 2pm." },
      {
        text: "Actually, change that to lip filler.",
        expectAnswerAny: ["lip", "filler", "0.5", "1 ml"]
      },
      {
        text: "0.5 ml please.",
        forbidAnswerAny: ["which massage", "massage duration"]
      }
    ]
  },
  {
    name: "price-side-question-during-flow",
    messages: [
      { text: "I'd like to book EMS tomorrow at 2pm." },
      {
        text: "How much is EMS?",
        expectAnswerAny: ["cost", "price", "200"],
        expectWorkflowContinuation: true
      },
      { text: "Ilko, 07700900114." }
    ]
  },
  {
    name: "schedule-without-contact-details",
    messages: [
      { text: "I'd like a one hour relaxing massage tomorrow at 2pm." }
    ],
    finalForbidAnswerAny: ["will be in touch", "will contact", "pass your request"]
  },
  {
    name: "detect-premature-handoff",
    messages: [
      {
        text: "Hey, I'd like to book.",
        expectQuestionTypes: ["treatment"]
      }
    ],
    finalForbidAnswerAny: ["let Veronika know", "will be in touch", "pass your request"]
  },
  {
    name: "tomorrow-at-two-is-understood",
    messages: [
      { text: "I'd like a one hour relaxing massage tomorrow at 2pm." },
      {
        text: "My name is Ilko and my number is 07700900115.",
        forbidQuestionTypes: ["date", "time"]
      }
    ]
  },
  {
    name: "short-replies-complete-flow",
    messages: [
      { text: "I want a massage." },
      { text: "deep tissue" },
      { text: "one hour" },
      { text: "tomorrow" },
      { text: "2pm" },
      { text: "Ilko" },
      { text: "07700900116" }
    ]
  },
  {
    name: "hot-stone-short-duration-and-plain-text",
    messages: [
      {
        text: "Do you offer massages?",
        expectAnswerAny: ["massage", "hot stone"]
      },
      {
        text: "How much is hot stone 30 minutes?",
        expectAnswerAny: ["hot stone", "60", "55"],
        expectQuestionTypes: ["duration"]
      },
      {
        text: "60 then",
        forbidQuestionTypes: ["duration"],
        expectQuestionTypes: ["date"]
      }
    ]
  }
];
