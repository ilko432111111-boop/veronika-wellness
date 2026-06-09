const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");
const scenarios = require("./scenarios");

const TAKE_SCREENSHOTS = false;

const ROOT = __dirname;
const SCREENSHOT_DIR = path.join(ROOT, "screenshots");
const REPORT_DIR = path.join(ROOT, "reports");
const JSON_REPORT = path.join(REPORT_DIR, "latest-report.json");
const TEXT_TRANSCRIPTS = path.join(REPORT_DIR, "latest-transcripts.txt");
const MARKDOWN_TRANSCRIPTS = path.join(REPORT_DIR, "latest-transcripts.md");
const CLI_ARGS = process.argv.slice(2);
const FAILURE_CATEGORIES = [
  "Connection / Infrastructure",
  "Calendar / Availability",
  "Lead Capture",
  "Treatment Selection",
  "Service Switching",
  "Side Question Handling",
  "Premature Handoff",
  "Soft Premature Handoff",
  "Contact Detail Ignored",
  "Repeated Slot Prompt",
  "State Persistence"
];

const BASE_URL = process.env.QA_BASE_URL || "https://veronika-wellness.onrender.com";
const HEADLESS = !CLI_ARGS.includes("--headed") &&
  String(process.env.QA_HEADLESS || "true").toLowerCase() !== "false";
const TIMEOUT = Number(process.env.QA_TIMEOUT || 90000);
const slowMoArgument = CLI_ARGS.find((argument) => argument.startsWith("--slow-mo="));
const SLOW_MO = Number(
  slowMoArgument ? slowMoArgument.split("=")[1] : process.env.QA_SLOW_MO || 0
);
const SCENARIO_FILTER = String(process.env.QA_SCENARIO || "").trim().toLowerCase();

const SELECTORS = {
  chatButton: "#chat-button",
  chatWindow: "#chat-window",
  input: "#chat-input",
  sendButton: "#chat-input-area button",
  userMessages: "#chat-messages .user-message",
  botMessages: "#chat-messages .bot-message",
  typingText: "Veronika's assistant is typing..."
};

const HANDOFF_PATTERNS = [
  /\b(?:i(?:'|\u2019)ll|i have|i(?:'|\u2019)ve)\s+let\s+veronika\s+know\b/i,
  /\b(?:i(?:'|\u2019)ll|we(?:'|\u2019)ll|i will|we will)\s+pass\s+(?:your\s+)?(?:request|details)\s+on\b/i,
  /\b(?:i(?:'|\u2019)ll|we(?:'|\u2019)ll|i will|we will)\s+(?:be in touch|get back to you|follow up)\b/i,
  /\bveronika\s+will\s+(?:be in touch|contact|review|get back|follow up|confirm)\b/i,
  /\b(?:she|the therapist)\s+will\s+(?:be in touch|contact|get back|follow up|confirm)\b/i,
  /\b(?:request|details|booking request)\s+(?:has|have)\s+been\s+(?:sent|passed on|forwarded)\b/i
];

const SOFT_HANDOFF_PATTERNS = [
  /\b(?:we|i)(?:'|\u2019)ll\s+take care of\b/i,
  /\bwe(?:'|\u2019)ll\s+get everything set up\b/i,
  /\bwe(?:'|\u2019)ll\s+make sure everything is ready\b/i,
  /\b(?:we|i)(?:'|\u2019)ll\s+arrange (?:the )?(?:remaining )?details\b/i,
  /\bi(?:'|\u2019)ll\s+look into suitable times and get back to you\b/i,
  /\bi(?:'|\u2019)ve noted your request and (?:i |we )?will take care of\b/i,
  /\bi(?:'|\u2019)ve noted your request and (?:i(?:'|\u2019)ll|we(?:'|\u2019)ll)\s+take care of\b/i,
  /\bi(?:'|\u2019)ve noted (?:your|the)\s+(?:request|preference|details?)[^.!?]{0,80}\band will take care of\b/i
];

const AUTONOMOUS_CONFIRMATION_PATTERNS = [
  /\byou(?:'|\u2019)re booked\b/i,
  /\byou are booked\b/i,
  /\bbooking is confirmed\b/i,
  /\bappointment is confirmed\b/i,
  /\bi(?:'|\u2019)ve booked you\b/i,
  /\bi have booked you\b/i,
  /\bslot is reserved\b/i,
  /\bpayment has been taken\b/i
];

const QUESTION_PATTERNS = {
  treatment: /\b(?:which|what)\s+(?:treatment|service|massage|facial|filler)\b/i,
  duration: /\b(?:how long|what duration|which duration|choose (?:a |your )?(?:session )?length|select (?:a |your )?(?:session )?length|what session length|which session length)\b/i,
  date: /\b(?:which|what)\s+(?:day|date)\b[^?]*\?/i,
  time: /\b(?:which|what)\s+time\b[^?]*\?/i,
  name: /\b(?:what(?:'s| is) your name|could i take your name|may i have your name|can i have your name|your name,?\s*please)\b/i,
  phone: /\b(?:what(?:'s| is) your (?:phone|mobile|contact) number|could i take your (?:phone|mobile|contact) number|may i have your (?:phone|mobile|contact) number|can i have your (?:phone|mobile|contact) number|your (?:phone|mobile|contact) number,?\s*please)\b/i
};

const DURATION_OPTION_REQUEST_PATTERNS = [
  /\b(?:would you like|do you prefer|which would you prefer|please choose|choose|select|pick)\b[^?.!]{0,100}\b(?:30|45|60|90|120)\s*(?:minutes?|mins?)\b/i,
  /\b(?:would you like|do you prefer|which would you prefer|please choose|choose|select|pick)\b[^?.!]{0,100}\b(?:one|two|1|2)\s*hours?\b/i,
  /\b(?:30|45|60|90|120)\s*(?:minutes?|mins?)\b[^?.!]{0,100}\b(?:which|what)\s+(?:duration|session length)\b/i,
  /\b30\s*\/\s*60\s*\/\s*90(?:\s*\/\s*120)?\s*(?:minutes?|mins?)?\s*\?/i
];

function ensureDirectories() {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  fs.mkdirSync(REPORT_DIR, { recursive: true });
}

function clearGeneratedOutputs() {
  const directories = TAKE_SCREENSHOTS ? [SCREENSHOT_DIR, REPORT_DIR] : [REPORT_DIR];
  for (const directory of directories) {
    for (const filename of fs.readdirSync(directory)) {
      if (filename === ".gitkeep") {
        continue;
      }
      fs.rmSync(path.join(directory, filename), { force: true });
    }
  }
}

function slugify(value) {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

function londonDate(offsetDays = 0) {
  const london = new Date(
    new Date().toLocaleString("en-US", { timeZone: "Europe/London" })
  );
  london.setDate(london.getDate() + offsetDays);
  return london.toISOString().slice(0, 10);
}

function createConversationState() {
  return {
    treatment: null,
    treatmentResolved: false,
    duration: null,
    preferredDate: null,
    preferredTime: null,
    name: null,
    phone: null,
    availabilityVerified: false,
    nameClearlyRequested: false,
    phoneClearlyRequested: false,
    lastBotReply: "",
    lastSlotOptionsSignature: null
  };
}

function updateStateFromUser(state, message) {
  const text = message.trim();
  const lower = text.toLowerCase();

  const treatments = [
    ["lip filler", "Lip Filler", false],
    ["deep tissue", "Deep Tissue Massage", true],
    ["relaxing massage", "Relaxing Massage", true],
    ["ems", "EMS", true],
    ["ultrasound", "Ultrasound", false],
    ["massage", "Massage", false],
    ["facial", "Facial", false]
  ];

  for (const [token, label, resolved] of treatments) {
    if (lower.includes(token)) {
      const treatmentChanged = state.treatment && state.treatment !== label;
      state.treatment = label;
      state.treatmentResolved = resolved;
      if (treatmentChanged) {
        state.duration = null;
      }
      if (label === "EMS") {
        state.duration = "service metadata duration";
      }
      state.availabilityVerified = false;
      break;
    }
  }

  if (/\b(?:0\.5|1)\s*ml\b/i.test(text) && /filler/i.test(state.treatment || "")) {
    state.treatmentResolved = true;
    state.duration = "service metadata duration";
  }

  const duration = lower.match(/\b(30|45|60|90|120)\s*minutes?\b|\b(one|1|two|2)\s*hours?\b/);
  if (duration) {
    state.duration = duration[0];
    state.availabilityVerified = false;
  }

  if (/\btoday\b/i.test(text)) {
    state.preferredDate = londonDate(0);
    state.availabilityVerified = false;
  } else if (/\btomorrow\b/i.test(text)) {
    state.preferredDate = londonDate(1);
    state.availabilityVerified = false;
  } else if (/\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i.test(text)) {
    state.preferredDate = "weekday-supplied";
    state.availabilityVerified = false;
  }

  const time = lower.match(/\b(?:[01]?\d|2[0-3]):[0-5]\d\b|\b(?:1[0-2]|[1-9])\s*(?:am|pm)\b/);
  if (time) {
    state.preferredTime = time[0];
    state.availabilityVerified = false;
  }

  const phone = text.match(/(?:\+44|0)7[\d\s-]{8,13}\d/);
  if (phone) {
    state.phone = phone[0].replace(/[^\d+]/g, "");
  }

  if (/\b(?:my name is|i am|i'm)\s+([a-z][a-z'-]+)/i.test(text)) {
    state.name = text.match(/\b(?:my name is|i am|i'm)\s+([a-z][a-z'-]+)/i)[1];
  } else if (asksQuestionType(state.lastBotReply, "name") && /^[a-z][a-z'-]+$/i.test(text)) {
    state.name = text;
  } else if (/^[A-Z][a-z'-]+$/.test(text)) {
    state.name = text;
  }

  if (
    !state.name &&
    state.phone &&
    /^[a-z][a-z'-]+\s*[,-]\s*(?:\+44|0)7/i.test(text)
  ) {
    state.name = text.match(/^([a-z][a-z'-]+)/i)[1];
  }
}

function contactDetailsProvided(message) {
  const text = message.trim();
  const phone = text.match(/(?:\+44|0)7[\d\s-]{8,13}\d/)?.[0]?.replace(/[^\d+]/g, "") || null;
  const explicitName = text.match(/\b(?:my name is|i am|i'm)\s+([a-z][a-z'-]+)/i)?.[1] ||
    text.match(/^([a-z][a-z'-]+)\s*[,-]\s*(?:\+44|0)7/i)?.[1] ||
    (/^[A-Z][a-z'-]+$/.test(text) ? text : null);
  return {
    providedName: Boolean(explicitName),
    providedPhone: Boolean(phone),
    name: explicitName || null,
    phone
  };
}

function updateStateFromBot(state, reply) {
  if (/\bcurrently appears free\b/i.test(reply)) {
    state.availabilityVerified = true;
  }
  if (/\b(?:could not verify|unavailable|closed|already passed)\b/i.test(reply)) {
    state.availabilityVerified = false;
  }
  state.nameClearlyRequested ||= asksQuestionType(reply, "name");
  state.phoneClearlyRequested ||= asksQuestionType(reply, "phone");
  state.lastSlotOptionsSignature = slotOptionsSignature(reply);
  state.lastBotReply = reply;
}

function asksQuestionType(reply, type) {
  if (type === "duration") {
    return asksForDuration(reply);
  }
  return Boolean(QUESTION_PATTERNS[type] && QUESTION_PATTERNS[type].test(reply));
}

function asksForDuration(reply) {
  return QUESTION_PATTERNS.duration.test(reply) ||
    DURATION_OPTION_REQUEST_PATTERNS.some((pattern) => pattern.test(reply));
}

function countQuestions(reply) {
  return (reply.match(/\?/g) || []).length;
}

function hasAny(text, values = []) {
  const lower = text.toLowerCase();
  return values.some((value) => lower.includes(String(value).toLowerCase()));
}

function handoffEligibleForVisibleConversation(state) {
  return Boolean(
    state.treatment &&
    state.treatmentResolved &&
    state.duration &&
    state.preferredDate &&
    state.preferredTime &&
    state.availabilityVerified &&
    (
      state.name && state.phone ||
      state.nameClearlyRequested && state.phoneClearlyRequested
    )
  );
}

function leadCompleteForProceeding(state) {
  return Boolean(
    state.treatment &&
    state.treatmentResolved &&
    state.duration &&
    state.preferredDate &&
    state.preferredTime &&
    state.availabilityVerified &&
    state.name &&
    state.phone
  );
}

function acknowledgesContactDetails(reply, turnContext) {
  const lower = reply.toLowerCase();
  if (/\b(?:thank you|thanks)\s+(?:for sharing|for providing)\s+(?:your\s+)?details\b/i.test(reply) ||
      /\b(?:noted|saved|got)\s+(?:your\s+)?(?:contact details|details|name|phone|number)\b/i.test(reply)) {
    return true;
  }
  if (turnContext.providedName && turnContext.name && lower.includes(turnContext.name.toLowerCase())) {
    return true;
  }
  return turnContext.providedPhone && turnContext.phone &&
    reply.replace(/[^\d]/g, "").includes(turnContext.phone.replace(/[^\d]/g, ""));
}

function asksClearMissingDetailQuestion(reply) {
  return ["treatment", "duration", "date", "time", "name", "phone"]
    .some((type) => asksQuestionType(reply, type));
}

function slotOptionsSignature(reply) {
  if (!/\b(?:next available options|current verified options|verified options)\b/i.test(reply)) {
    return null;
  }
  return reply
    .split(/\n\s*\n/)[0]
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function issueGuidance(message) {
  if (/soft premature handoff/i.test(message)) {
    return {
      likelyRootCause: "The natural responder is adding reassuring proceed-language before the lead is complete.",
      suggestedFixDirection: "Strip soft proceed-language during active collection and allow it only after treatment, schedule, verified availability, name, and phone are resolved."
    };
  }
  if (/contact detail ignored/i.test(message)) {
    return {
      likelyRootCause: "The latest contact-detail extraction is not influencing final response composition.",
      suggestedFixDirection: "Acknowledge newly supplied contact details before continuing with one clear next-required-detail question."
    };
  }
  if (/repeated slot prompt/i.test(message)) {
    return {
      likelyRootCause: "Pending slot alternatives are overriding the customer's latest useful information.",
      suggestedFixDirection: "Process and acknowledge the latest message before repeating verified alternatives, and avoid repeating an unchanged slot prompt verbatim."
    };
  }
  if (/premature handoff|handoff wording/i.test(message)) {
    return {
      likelyRootCause: "Handoff language is being generated before the workflow state passes the handoff eligibility gate.",
      suggestedFixDirection: "Keep handoff wording Python-controlled and append it only after every required detail and backend verification are complete."
    };
  }
  if (/autonomously confirmed|reserved/i.test(message)) {
    return {
      likelyRootCause: "The natural responder is allowed to imply confirmation instead of using backend-controlled booking status wording.",
      suggestedFixDirection: "Strip autonomous confirmation claims and render only verified, Python-controlled availability or handoff wording."
    };
  }
  if (/availability claim/i.test(message)) {
    return {
      likelyRootCause: "The reply generator is treating service information or unverified schedule context as confirmed slot availability.",
      suggestedFixDirection: "Require a verified backend availability result before allowing slot-availability claims."
    };
  }
  if (/asked for .+ even though|repeated a question/i.test(message)) {
    return {
      likelyRootCause: "The workflow question selector is not consistently reading or preserving already-collected canonical state.",
      suggestedFixDirection: "Select the next workflow question from saved canonical state and skip fields that are already resolved."
    };
  }
  if (/informational answer|side-question answer/i.test(message)) {
    return {
      likelyRootCause: "The responder is prioritising the pending booking workflow over the customer's latest informational intent.",
      suggestedFixDirection: "Detect and answer informational side questions before appending at most one Python-controlled workflow question."
    };
  }
  if (/forbidden stale content/i.test(message)) {
    return {
      likelyRootCause: "The responder or workflow state is still using details from an earlier customer choice.",
      suggestedFixDirection: "Invalidate stale dependent state when the customer changes details, then compose the reply from the latest canonical values."
    };
  }
  if (/expected the bot to ask|final reply did not request/i.test(message)) {
    return {
      likelyRootCause: "The next-required-detail selector did not choose the expected missing workflow field.",
      suggestedFixDirection: "Derive the next question deterministically from unresolved canonical state after processing the latest message."
    };
  }
  if (/more than one workflow question/i.test(message)) {
    return {
      likelyRootCause: "Multiple response layers are appending workflow questions independently.",
      suggestedFixDirection: "Make final composition append exactly one Python-controlled next question and strip model-generated workflow questions."
    };
  }
  if (/empty reply|punctuation-only|meaningless fallback/i.test(message)) {
    return {
      likelyRootCause: "Reply sanitization removed the useful response without producing a meaningful deterministic fallback.",
      suggestedFixDirection: "After sanitization, detect meaningless output and render an authoritative answer or the next required workflow question."
    };
  }
  if (/runner error/i.test(message)) {
    return {
      likelyRootCause: "The browser run could not reach or complete interaction with the live chatbot.",
      suggestedFixDirection: "Check live-site reachability, service health, and runner timeout settings, then rerun the affected scenario."
    };
  }
  return {
    likelyRootCause: "The final reply did not match the expected receptionist-flow contract.",
    suggestedFixDirection: "Trace the latest-message intent, canonical workflow state, and final response composition for this turn."
  };
}

function issueImpact(message) {
  if (/soft premature handoff/i.test(message)) {
    return "The customer may believe the business is already proceeding with an incomplete or unverified request.";
  }
  if (/contact detail ignored/i.test(message)) {
    return "The customer cannot tell whether their name or phone number was captured, which undermines lead capture reliability.";
  }
  if (/repeated slot prompt/i.test(message)) {
    return "Repeating unchanged slot options after useful new information makes the bot appear stuck and ignores customer progress.";
  }
  if (/premature handoff|handoff wording/i.test(message)) {
    return "The customer may believe their incomplete request has already been handed to Veronika.";
  }
  if (/autonomously confirmed|reserved/i.test(message)) {
    return "The customer may believe an appointment is confirmed or reserved when the receptionist flow cannot guarantee that.";
  }
  if (/availability claim/i.test(message)) {
    return "The customer may rely on a slot that has not been verified by the backend.";
  }
  if (/asked for .+ even though|repeated a question/i.test(message)) {
    return "Repeating an already-answered question makes the bot appear to have lost customer details and can stall the flow.";
  }
  if (/informational answer/i.test(message)) {
    return "The customer asked a side question, but the reply did not answer it.";
  }
  if (/side-question answer/i.test(message)) {
    return "The side question did not cleanly return the customer to the active booking flow.";
  }
  if (/forbidden stale content/i.test(message)) {
    return "The reply uses outdated details after the customer changed their request.";
  }
  if (/expected the bot to ask|final reply did not request/i.test(message)) {
    return "The booking flow cannot progress because the bot did not request the next required detail.";
  }
  if (/more than one workflow question/i.test(message)) {
    return "Multiple workflow questions make the next required customer action ambiguous.";
  }
  if (/empty reply|punctuation-only|meaningless fallback/i.test(message)) {
    return "The customer receives no useful answer or next action.";
  }
  if (/runner error/i.test(message)) {
    return "The scenario could not be evaluated, so the report cannot confirm whether the chatbot flow passed.";
  }
  return "The reply does not satisfy the expected receptionist-flow behavior for this turn.";
}

function failureCategory(message, expected = "") {
  const combined = `${message} ${expected}`;
  if (/runner error|connection|network|timeout|reach|service health/i.test(combined)) {
    return "Connection / Infrastructure";
  }
  if (/soft premature handoff/i.test(combined)) {
    return "Soft Premature Handoff";
  }
  if (/contact detail ignored/i.test(combined)) {
    return "Contact Detail Ignored";
  }
  if (/repeated slot prompt/i.test(combined)) {
    return "Repeated Slot Prompt";
  }
  if (/premature handoff|handoff wording|autonomously confirmed|reserved/i.test(combined)) {
    return "Premature Handoff";
  }
  if (/side-question|informational answer/i.test(combined)) {
    return "Side Question Handling";
  }
  if (/forbidden stale content|changed|latest supplied details/i.test(combined)) {
    return "Service Switching";
  }
  if (/availability|calendar|slot|schedule|verified backend/i.test(combined)) {
    return "Calendar / Availability";
  }
  if (/treatment|service detail|service choice|duration|variant|session length/i.test(combined)) {
    return "Treatment Selection";
  }
  if (/\b(?:name|phone|mobile|contact number|customer detail)\b/i.test(combined)) {
    return "Lead Capture";
  }
  return "State Persistence";
}

function createIssue({
  confidence,
  message,
  expected,
  actual,
  reason,
  likelyRootCause,
  suggestedFixDirection,
  category
}) {
  const guidance = issueGuidance(message);
  return {
    confidence: formatConfidence(confidence),
    category: category || failureCategory(message, expected),
    message,
    expected,
    actual,
    whyProblem: issueImpact(message),
    detectionReason: reason,
    likelyRootCause: likelyRootCause || guidance.likelyRootCause,
    suggestedFixDirection: suggestedFixDirection || guidance.suggestedFixDirection
  };
}

function detectIssues({ state, reply, step, turnContext = {} }) {
  const issues = [];
  const handoffDetected = HANDOFF_PATTERNS.some((pattern) => pattern.test(reply));
  const softHandoffDetected = SOFT_HANDOFF_PATTERNS.some((pattern) => pattern.test(reply));

  if (handoffDetected && !handoffEligibleForVisibleConversation(state)) {
    issues.push(createIssue({
      confidence: "definite failure",
      message: "Premature handoff wording appeared before the visible conversation satisfied the required flow.",
      expected: "Continue collecting required booking details without implying handoff.",
      actual: reply,
      reason: "The reply used handoff wording before the visible conversation met handoff eligibility."
    }));
  }

  if (softHandoffDetected && !leadCompleteForProceeding(state)) {
    issues.push(createIssue({
      confidence: "definite failure",
      category: "Soft Premature Handoff",
      message: "Soft premature handoff wording appeared before the lead was complete.",
      expected: "Continue collecting treatment, duration, date, time, verified availability, name, and phone without implying the business will proceed.",
      actual: reply,
      reason: "The reply implied the business would take care of remaining steps before the complete lead state was resolved."
    }));
  }

  const contactProvided = turnContext.providedName || turnContext.providedPhone;
  const contactAcknowledged = contactProvided && acknowledgesContactDetails(reply, turnContext);
  const currentSlotSignature = slotOptionsSignature(reply);
  if (
    contactProvided &&
    !contactAcknowledged &&
    !asksClearMissingDetailQuestion(reply)
  ) {
    issues.push(createIssue({
      confidence: "likely failure",
      category: "Contact Detail Ignored",
      message: "Contact detail ignored after the customer supplied a name or phone number.",
      expected: "Acknowledge the newly supplied contact detail or continue with one clear missing-detail question.",
      actual: reply,
      reason: "The reply neither acknowledged the new contact information nor asked a clear next-required-detail question."
    }));
  }

  if (
    contactProvided &&
    !contactAcknowledged &&
    currentSlotSignature &&
    currentSlotSignature === turnContext.previousSlotOptionsSignature
  ) {
    issues.push(createIssue({
      confidence: "likely failure",
      category: "Repeated Slot Prompt",
      message: "Repeated slot prompt ignored useful contact details.",
      expected: "Acknowledge the newly supplied name or phone before repeating or advancing slot selection.",
      actual: reply,
      reason: "The bot repeated the same verified options after the customer supplied useful contact information."
    }));
  }

  if (AUTONOMOUS_CONFIRMATION_PATTERNS.some((pattern) => pattern.test(reply))) {
    issues.push(createIssue({
      confidence: "definite failure",
      message: "The bot implied that the booking was autonomously confirmed or reserved.",
      expected: "Act as a receptionist and do not confirm or reserve appointments autonomously.",
      actual: reply,
      reason: "The reply matched explicit autonomous confirmation or reservation wording."
    }));
  }

  const availabilityClaim = /\b(?:currently appears free|slot is free|appointment is available|time is available|can fit you in|availability (?:on|at))\b/i.test(reply);
  if (
    availabilityClaim &&
    !/currently appears free/i.test(reply) &&
    !(state.treatment && state.preferredDate && state.preferredTime)
  ) {
    issues.push(createIssue({
      confidence: "likely failure",
      message: "The bot made an availability claim before enough schedule details were supplied.",
      expected: "Only claim slot availability after treatment, date, and time are known and backend verification has run.",
      actual: reply,
      reason: "The reply used slot-availability wording before the visible conversation supplied enough schedule details."
    }));
  }

  const alreadyKnown = {
    treatment: state.treatmentResolved,
    duration: Boolean(state.duration),
    date: Boolean(state.preferredDate),
    time: Boolean(state.preferredTime),
    name: Boolean(state.name),
    phone: Boolean(state.phone)
  };
  for (const [type, known] of Object.entries(alreadyKnown)) {
    if (known && asksQuestionType(reply, type)) {
      issues.push(createIssue({
        confidence: "definite failure",
        message: `The bot asked for ${type} even though it was already supplied.`,
        expected: `${type[0].toUpperCase()}${type.slice(1)} already known.`,
        actual: reply,
        reason: `Bot explicitly requested ${type} again.`
      }));
    }
  }

  if (countQuestions(reply) > 1) {
    issues.push(createIssue({
      confidence: "likely failure",
      message: "The reply contains more than one workflow question.",
      expected: "Ask at most one workflow question in the final reply.",
      actual: reply,
      reason: "The reply contains more than one question mark."
    }));
  }

  if (step.expectAnswerAny && !hasAny(reply, step.expectAnswerAny)) {
    issues.push(createIssue({
      confidence: "definite failure",
      message: `Expected an informational answer containing one of: ${step.expectAnswerAny.join(", ")}.`,
      expected: `Answer the side question using one of these expected concepts: ${step.expectAnswerAny.join(", ")}.`,
      actual: reply,
      reason: "The reply did not contain any expected answer concept."
    }));
  }
  if (step.expectWorkflowContinuation && countQuestions(reply) !== 1) {
    issues.push(createIssue({
      confidence: "likely failure",
      message: "Expected the side-question answer to continue with exactly one workflow question.",
      expected: "Answer the side question and continue with exactly one workflow question.",
      actual: reply,
      reason: `The reply contained ${countQuestions(reply)} questions.`
    }));
  }
  if (step.forbidAnswerAny && hasAny(reply, step.forbidAnswerAny)) {
    issues.push(createIssue({
      confidence: "definite failure",
      message: `Reply contained forbidden stale content: ${step.forbidAnswerAny.join(", ")}.`,
      expected: "Continue using the customer's latest supplied details.",
      actual: reply,
      reason: "The reply contained content explicitly forbidden by the scenario."
    }));
  }
  for (const type of step.expectQuestionTypes || []) {
    if (!asksQuestionType(reply, type)) {
      issues.push(createIssue({
        confidence: "definite failure",
        message: `Expected the bot to ask for ${type}.`,
        expected: `Ask for ${type}.`,
        actual: reply,
        reason: `The reply did not explicitly request ${type}.`
      }));
    }
  }
  for (const type of step.forbidQuestionTypes || []) {
    if (asksQuestionType(reply, type)) {
      issues.push(createIssue({
        confidence: "definite failure",
        message: `The bot repeated a question for already supplied ${type}.`,
        expected: `${type[0].toUpperCase()}${type.slice(1)} already known.`,
        actual: reply,
        reason: `Bot explicitly requested ${type} again.`
      }));
    }
  }

  if (!reply.trim()) {
    issues.push(createIssue({
      confidence: "definite failure",
      message: "The bot returned an empty reply.",
      expected: "Return a meaningful receptionist response.",
      actual: reply,
      reason: "The reply was empty."
    }));
  }
  if (/^[\s\p{P}]+$/u.test(reply)) {
    issues.push(createIssue({
      confidence: "definite failure",
      message: "The bot returned punctuation-only content.",
      expected: "Return a meaningful receptionist response.",
      actual: reply,
      reason: "The reply contained only whitespace or punctuation."
    }));
  }
  if (reply.trim().toLowerCase() === "thanks.") {
    issues.push(createIssue({
      confidence: "definite failure",
      message: 'The bot returned the meaningless fallback "Thanks.".',
      expected: "Return a meaningful answer or workflow question.",
      actual: reply,
      reason: 'The sole reply was "Thanks.".'
    }));
  }

  return issues.filter((issue, index) =>
    issues.findIndex((candidate) => candidate.message === issue.message) === index
  );
}

async function takeScreenshot(page, scenarioSlug, stepNumber, phase) {
  if (!TAKE_SCREENSHOTS) {
    return null;
  }

  const filename = `${scenarioSlug}-${String(stepNumber).padStart(2, "0")}-${phase}.png`;
  const fullPath = path.join(SCREENSHOT_DIR, filename);
  await page.screenshot({ path: fullPath, fullPage: true });
  return path.relative(ROOT, fullPath).replace(/\\/g, "/");
}

function writeAnnotation(scenarioSlug, stepNumber, issues, screenshot) {
  if (!TAKE_SCREENSHOTS || !screenshot) {
    return null;
  }

  const filename = `${scenarioSlug}-${String(stepNumber).padStart(2, "0")}-issues.txt`;
  const fullPath = path.join(SCREENSHOT_DIR, filename);
  const body = [
    `Screenshot: ${screenshot}`,
    "",
    "Detected issues:",
    ...issues.flatMap((issue) => [
      `- [${issue.confidence}] ${issue.message}`,
      `  Failure category: ${issue.category}`,
      `  Expected: ${issue.expected}`,
      `  Actual: ${issue.actual}`,
      `  Why this is a problem: ${issue.whyProblem}`,
      `  Detection reason: ${issue.detectionReason}`,
      `  Likely root cause: ${issue.likelyRootCause}`,
      `  Suggested fix direction: ${issue.suggestedFixDirection}`
    ])
  ].join("\n");
  fs.writeFileSync(fullPath, body, "utf8");
  return path.relative(ROOT, fullPath).replace(/\\/g, "/");
}

async function waitForBotReply(page, previousBotCount) {
  const botMessages = page.locator(SELECTORS.botMessages);
  const typing = page.getByText(SELECTORS.typingText, { exact: true });

  await typing.waitFor({ state: "visible", timeout: 10000 }).catch(() => {});
  await page.waitForFunction(
    ({ selector, previous, typingText }) => {
      const messages = Array.from(document.querySelectorAll(selector));
      const finalMessages = messages.filter(
        (message) => message.textContent.trim() !== typingText
      );
      return finalMessages.length > previous;
    },
    {
      selector: SELECTORS.botMessages,
      previous: previousBotCount,
      typingText: SELECTORS.typingText
    },
    { timeout: TIMEOUT }
  );
  await typing.waitFor({ state: "detached", timeout: TIMEOUT }).catch(() => {});

  const finalMessages = botMessages.filter({
    hasNotText: SELECTORS.typingText
  });
  return (await finalMessages.last().innerText()).trim();
}

async function runScenario(browser, scenario) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1100 },
    locale: "en-GB",
    timezoneId: "Europe/London"
  });
  const page = await context.newPage();
  const scenarioSlug = slugify(scenario.name);
  const state = createConversationState();
  const result = {
    testName: scenario.name,
    url: BASE_URL,
    status: "passed",
    startedAt: new Date().toISOString(),
    conversation: [],
    detectedIssues: []
  };

  try {
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded", timeout: TIMEOUT });
    await page.locator(SELECTORS.chatButton).waitFor({ state: "visible", timeout: TIMEOUT });
    await page.locator(SELECTORS.chatButton).click();
    await page.locator(SELECTORS.chatWindow).waitFor({ state: "visible", timeout: TIMEOUT });

    for (let index = 0; index < scenario.messages.length; index += 1) {
      const step = scenario.messages[index];
      const stepNumber = index + 1;
      const botMessages = page.locator(SELECTORS.botMessages);
      const previousBotCount = await botMessages.evaluateAll((nodes, typingText) =>
        nodes.filter((node) => node.textContent.trim() !== typingText).length,
        SELECTORS.typingText
      );

      const turnContext = {
        ...contactDetailsProvided(step.text),
        previousSlotOptionsSignature: state.lastSlotOptionsSignature
      };
      updateStateFromUser(state, step.text);
      await page.locator(SELECTORS.input).fill(step.text);
      await page.locator(SELECTORS.sendButton).click();
      await page.locator(SELECTORS.userMessages).last().waitFor({ state: "visible" });
      const userScreenshot = await takeScreenshot(page, scenarioSlug, stepNumber, "user");

      const botReply = await waitForBotReply(page, previousBotCount);
      updateStateFromBot(state, botReply);
      const botScreenshot = await takeScreenshot(page, scenarioSlug, stepNumber, "bot");
      const issues = detectIssues({ state, reply: botReply, step, turnContext });
      let annotationFile = null;

      if (issues.length) {
        annotationFile = writeAnnotation(scenarioSlug, stepNumber, issues, botScreenshot);
        result.detectedIssues.push(
          ...issues.map((issue) => ({ step: stepNumber, ...issue }))
        );
      }

      result.conversation.push({
        step: stepNumber,
        userMessage: step.text,
        botReply,
        screenshots: {
          afterUserMessage: userScreenshot,
          afterBotReply: botScreenshot
        },
        annotationFile,
        detectedIssues: issues
      });
    }

    const finalReply = result.conversation.at(-1)?.botReply || "";
    const finalIssues = [];
    if (scenario.finalExpectQuestionAny && !hasAny(finalReply, scenario.finalExpectQuestionAny)) {
      finalIssues.push(createIssue({
        confidence: "definite failure",
        message: `Final reply did not request one of: ${scenario.finalExpectQuestionAny.join(", ")}.`,
        expected: `Request one of: ${scenario.finalExpectQuestionAny.join(", ")}.`,
        actual: finalReply,
        reason: "The final reply did not contain any expected request concept."
      }));
    }
    if (scenario.finalForbidAnswerAny && hasAny(finalReply, scenario.finalForbidAnswerAny)) {
      finalIssues.push(createIssue({
        confidence: "definite failure",
        message: `Final reply contained forbidden wording: ${scenario.finalForbidAnswerAny.join(", ")}.`,
        expected: "Do not use forbidden handoff or stale workflow wording.",
        actual: finalReply,
        reason: "The final reply contained wording explicitly forbidden by the scenario."
      }));
    }
    if (finalIssues.length && result.conversation.length) {
      const finalTurn = result.conversation.at(-1);
      finalTurn.detectedIssues.push(...finalIssues);
      result.detectedIssues.push(
        ...finalIssues.map((issue) => ({
          step: scenario.messages.length,
          ...issue
        }))
      );
      finalTurn.annotationFile = writeAnnotation(
        scenarioSlug,
        scenario.messages.length,
        finalTurn.detectedIssues,
        finalTurn.screenshots.afterBotReply
      );
    }
  } catch (error) {
    result.detectedIssues.push({
      step: result.conversation.length + 1,
      ...createIssue({
        confidence: "likely failure",
        message: `Runner error: ${error.message}`,
        expected: "Complete the scenario and collect all bot replies.",
        actual: error.message,
        reason: "The runner could not complete the scenario, so chatbot behavior could not be fully evaluated."
      })
    });
    const failureScreenshot = await takeScreenshot(
      page,
      scenarioSlug,
      result.conversation.length + 1,
      "runner-error"
    ).catch(() => null);
    if (failureScreenshot) {
      const runnerIssue = result.detectedIssues.at(-1);
      writeAnnotation(
        scenarioSlug,
        result.conversation.length + 1,
        [runnerIssue],
        failureScreenshot
      );
    }
  } finally {
    result.finishedAt = new Date().toISOString();
    result.status = result.detectedIssues.some(issueFailsScenario) ? "failed" : "passed";
    await context.close();
  }

  return result;
}

function scenarioSummary(scenario) {
  const failures = scenario.detectedIssues.filter(issueFailsScenario);
  const warnings = scenario.detectedIssues.filter((issue) => !issueFailsScenario(issue));
  if (!failures.length && !warnings.length) {
    return "The conversation completed without any detected receptionist-flow issues.";
  }

  const issueCount = failures.length;
  const affectedSteps = [...new Set(scenario.detectedIssues.map((issue) => issue.step))]
    .filter(Boolean)
    .join(", ");
  const failureSummary = issueCount
    ? `${issueCount} failure${issueCount === 1 ? " was" : "s were"} detected` +
      `${affectedSteps ? ` after message${affectedSteps.includes(",") ? "s" : ""} ${affectedSteps}` : ""}.`
    : "No failures were detected.";
  return `${failureSummary}${warnings.length ? ` ${warnings.length} informational warning${warnings.length === 1 ? " was" : "s were"} also recorded.` : ""}`;
}

function formatIssue(issue) {
  return `${issue.message} (after message ${issue.step})`;
}

function formatConfidence(confidence) {
  return confidence.replace(/\s+/g, "_");
}

function issueFailsScenario(issue) {
  return issue.confidence === "definite_failure" || issue.confidence === "likely_failure";
}

function appendPlainTextIssue(lines, issue, issueNumber) {
  lines.push(`ISSUE ${issueNumber}`);
  lines.push(`Confidence: ${formatConfidence(issue.confidence)}`);
  lines.push(`Failure category: ${issue.category}`);
  lines.push(`Finding: ${formatIssue(issue)}`);
  lines.push("Expected:");
  lines.push(issue.expected);
  lines.push("");
  lines.push("Actual bot reply:");
  lines.push(`"${String(issue.actual || "").replace(/"/g, '\\"')}"`);
  lines.push("");
  lines.push("Why this is a problem:");
  lines.push(issue.whyProblem);
  lines.push("");
  lines.push("Likely root cause:");
  lines.push(issue.likelyRootCause);
  lines.push("");
  lines.push("Suggested fix direction:");
  lines.push(issue.suggestedFixDirection);
  lines.push("");
}

function appendMarkdownIssue(lines, issue, issueNumber) {
  lines.push(`#### Issue ${issueNumber}`);
  lines.push("");
  lines.push(`**Confidence:** \`${formatConfidence(issue.confidence)}\``);
  lines.push("");
  lines.push(`**Failure category:** ${issue.category}`);
  lines.push("");
  lines.push(`**Finding:** ${formatIssue(issue)}`);
  lines.push("");
  lines.push("**Expected:**");
  lines.push("");
  lines.push(issue.expected);
  lines.push("");
  lines.push("**Actual bot reply:**");
  lines.push("");
  lines.push("```text");
  lines.push(String(issue.actual || ""));
  lines.push("```");
  lines.push("");
  lines.push("**Why this is a problem:**");
  lines.push("");
  lines.push(issue.whyProblem);
  lines.push("");
  lines.push("**Likely root cause:**");
  lines.push("");
  lines.push(issue.likelyRootCause);
  lines.push("");
  lines.push("**Suggested fix direction:**");
  lines.push("");
  lines.push(issue.suggestedFixDirection);
  lines.push("");
}

function textTranscripts(report) {
  const lines = [
    "VERONIKA WELLNESS CHATBOT QA TRANSCRIPTS",
    `RUN STARTED: ${report.startedAt}`,
    `BASE URL: ${report.baseUrl}`,
    `TIMEZONE: ${report.timezone}`,
    `PASSED: ${report.summary.passed}`,
    `FAILED: ${report.summary.failed}`,
    "",
    "FAILURE CATEGORIES:",
    ...FAILURE_CATEGORIES.map((category) => `[ ] ${category}`),
    ""
  ];

  for (const scenario of report.scenarios) {
    lines.push("==============================");
    lines.push(`SCENARIO: ${scenario.testName}`);
    lines.push(`STATUS: ${scenario.status.toUpperCase()}`);
    lines.push("ISSUES:");
    if (scenario.detectedIssues.length) {
      for (const [index, issue] of scenario.detectedIssues.entries()) {
        appendPlainTextIssue(lines, issue, index + 1);
      }
    } else {
      lines.push("- None");
    }
    lines.push("");
    lines.push("CONVERSATION:");
    if (scenario.conversation.length) {
      for (const turn of scenario.conversation) {
        lines.push(`USER: ${turn.userMessage}`);
        lines.push(`BOT: ${turn.botReply}`);
        lines.push("");
      }
    } else {
      lines.push("No conversation turns completed.");
      lines.push("");
    }
    lines.push("SUMMARY:");
    lines.push(scenarioSummary(scenario));
    lines.push("==============================");
    lines.push("");
  }

  lines.push("FINAL SUMMARY:");
  lines.push(
    `${report.summary.passed} scenario${report.summary.passed === 1 ? "" : "s"} passed; ` +
    `${report.summary.failed} scenario${report.summary.failed === 1 ? "" : "s"} failed.`
  );
  return `${lines.join("\n")}\n`;
}

function markdownTranscripts(report) {
  const lines = [
    "# Veronika Wellness Chatbot QA Transcripts",
    "",
    `- Run started: ${report.startedAt}`,
    `- Base URL: ${report.baseUrl}`,
    `- Timezone: ${report.timezone}`,
    `- Result: **${report.summary.failed ? "FAILED" : "PASSED"}**`,
    `- Scenarios: ${report.summary.total}`,
    `- Passed: ${report.summary.passed}`,
    `- Failed: ${report.summary.failed}`,
    "",
    "## Failure Categories",
    "",
    ...FAILURE_CATEGORIES.map((category) => `- [ ] ${category}`),
    ""
  ];

  for (const scenario of report.scenarios) {
    lines.push("---");
    lines.push("");
    lines.push(`## Scenario: ${scenario.testName}`);
    lines.push("");
    lines.push(`Status: **${scenario.status.toUpperCase()}**`);
    lines.push("");
    lines.push("### Issues");
    lines.push("");
    if (scenario.detectedIssues.length) {
      for (const [index, issue] of scenario.detectedIssues.entries()) {
        appendMarkdownIssue(lines, issue, index + 1);
      }
    } else {
      lines.push("- None");
    }
    lines.push("");
    lines.push("### Conversation");
    lines.push("");

    for (const turn of scenario.conversation) {
      lines.push(`**USER:** ${turn.userMessage}`);
      lines.push("");
      lines.push(`**BOT:** ${turn.botReply}`);
      lines.push("");
      if (turn.screenshots.afterUserMessage) {
        lines.push(`- User screenshot: \`${turn.screenshots.afterUserMessage}\``);
      }
      if (turn.screenshots.afterBotReply) {
        lines.push(`- Bot screenshot: \`${turn.screenshots.afterBotReply}\``);
      }
      if (turn.annotationFile) {
        lines.push(`- Annotation: \`${turn.annotationFile}\``);
      }
    }

    lines.push("### Summary");
    lines.push("");
    lines.push(scenarioSummary(scenario));
    lines.push("");
  }

  lines.push("---");
  lines.push("");
  lines.push("## Final Summary");
  lines.push("");
  lines.push(
    `${report.summary.passed} scenario${report.summary.passed === 1 ? "" : "s"} passed; ` +
    `${report.summary.failed} scenario${report.summary.failed === 1 ? "" : "s"} failed.`
  );
  lines.push("");
  return `${lines.join("\n")}\n`;
}

async function main() {
  ensureDirectories();
  clearGeneratedOutputs();
  const selectedScenarios = scenarios.filter(
    (scenario) => !SCENARIO_FILTER || scenario.name.toLowerCase().includes(SCENARIO_FILTER)
  );

  if (!selectedScenarios.length) {
    throw new Error(`No scenario matched QA_SCENARIO=${SCENARIO_FILTER}`);
  }

  const report = {
    startedAt: new Date().toISOString(),
    baseUrl: BASE_URL,
    timezone: "Europe/London",
    screenshotsEnabled: TAKE_SCREENSHOTS,
    failureCategories: FAILURE_CATEGORIES,
    scenarios: []
  };
  const browser = await chromium.launch({ headless: HEADLESS, slowMo: SLOW_MO });

  try {
    for (const scenario of selectedScenarios) {
      console.log(`Running: ${scenario.name}`);
      report.scenarios.push(await runScenario(browser, scenario));
    }
  } finally {
    await browser.close();
  }

  report.finishedAt = new Date().toISOString();
  report.summary = {
    total: report.scenarios.length,
    passed: report.scenarios.filter((scenario) => scenario.status === "passed").length,
    failed: report.scenarios.filter((scenario) => scenario.status === "failed").length
  };

  fs.writeFileSync(JSON_REPORT, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  fs.writeFileSync(TEXT_TRANSCRIPTS, textTranscripts(report), "utf8");
  fs.writeFileSync(MARKDOWN_TRANSCRIPTS, markdownTranscripts(report), "utf8");

  console.log("");
  console.log("QA run complete");
  console.log(`Passed: ${report.summary.passed}`);
  console.log(`Failed: ${report.summary.failed}`);
  console.log(`Text transcripts: ${TEXT_TRANSCRIPTS}`);
  console.log(`Markdown transcripts: ${MARKDOWN_TRANSCRIPTS}`);
  console.log(`JSON report: ${JSON_REPORT}`);
  process.exitCode = report.summary.failed ? 1 : 0;
}

main().catch((error) => {
  ensureDirectories();
  const failure = {
    startedAt: new Date().toISOString(),
    baseUrl: BASE_URL,
    timezone: "Europe/London",
    screenshotsEnabled: TAKE_SCREENSHOTS,
    failureCategories: FAILURE_CATEGORIES,
    summary: { total: 0, passed: 0, failed: 1 },
    runnerError: error.stack || error.message,
    scenarios: []
  };
  fs.writeFileSync(JSON_REPORT, `${JSON.stringify(failure, null, 2)}\n`, "utf8");
  fs.writeFileSync(
    TEXT_TRANSCRIPTS,
    `VERONIKA WELLNESS CHATBOT QA TRANSCRIPTS\n\nRUNNER FAILED:\n${failure.runnerError}\n`,
    "utf8"
  );
  fs.writeFileSync(
    MARKDOWN_TRANSCRIPTS,
    `# Veronika Wellness Chatbot QA Transcripts\n\n## Runner Failed\n\n\`\`\`\n${failure.runnerError}\n\`\`\`\n`,
    "utf8"
  );
  console.error(error);
  console.log("");
  console.log("QA runner failed");
  console.log("Passed: 0");
  console.log("Failed: 1");
  console.log(`Text transcripts: ${TEXT_TRANSCRIPTS}`);
  console.log(`Markdown transcripts: ${MARKDOWN_TRANSCRIPTS}`);
  console.log(`JSON report: ${JSON_REPORT}`);
  process.exitCode = 1;
});
