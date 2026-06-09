const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");
const scenarios = require("./scenarios");

const ROOT = __dirname;
const SCREENSHOT_DIR = path.join(ROOT, "screenshots");
const REPORT_DIR = path.join(ROOT, "reports");
const JSON_REPORT = path.join(REPORT_DIR, "latest-report.json");
const MARKDOWN_REPORT = path.join(REPORT_DIR, "latest-report.md");
const CLI_ARGS = process.argv.slice(2);

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
  /\bveronika\s+will\s+(?:be in touch|contact|review|get back|follow up|confirm)\b/i,
  /\b(?:she|the therapist)\s+will\s+(?:be in touch|contact|get back|follow up|confirm)\b/i,
  /\b(?:request|details|booking request)\s+(?:has|have)\s+been\s+(?:sent|passed on|forwarded)\b/i
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
  duration: /\b(?:how long|which duration|what duration|30|45|60|90|120 minutes)\b[^?]*\?/i,
  date: /\b(?:which|what)\s+(?:day|date)\b[^?]*\?/i,
  time: /\b(?:which|what)\s+time\b[^?]*\?/i,
  name: /\b(?:your name|take your name)\b[^?]*\?/i,
  phone: /\b(?:phone|mobile|contact number)\b[^?]*\?/i
};

function ensureDirectories() {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  fs.mkdirSync(REPORT_DIR, { recursive: true });
}

function clearGeneratedOutputs() {
  for (const directory of [SCREENSHOT_DIR, REPORT_DIR]) {
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
    lastBotReply: ""
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
  }

  if (
    !state.name &&
    state.phone &&
    /^[a-z][a-z'-]+\s*[,-]\s*(?:\+44|0)7/i.test(text)
  ) {
    state.name = text.match(/^([a-z][a-z'-]+)/i)[1];
  }
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
  state.lastBotReply = reply;
}

function asksQuestionType(reply, type) {
  return Boolean(QUESTION_PATTERNS[type] && QUESTION_PATTERNS[type].test(reply));
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

function detectIssues({ state, reply, step }) {
  const issues = [];
  const handoffDetected = HANDOFF_PATTERNS.some((pattern) => pattern.test(reply));

  if (handoffDetected && !handoffEligibleForVisibleConversation(state)) {
    issues.push(
      "Premature handoff wording appeared before the visible conversation satisfied the required flow."
    );
  }

  if (AUTONOMOUS_CONFIRMATION_PATTERNS.some((pattern) => pattern.test(reply))) {
    issues.push("The bot implied that the booking was autonomously confirmed or reserved.");
  }

  const availabilityClaim = /\b(?:available|currently appears free|slot is free|fit you in)\b/i.test(reply);
  if (
    availabilityClaim &&
    !/currently appears free/i.test(reply) &&
    !(state.treatment && state.preferredDate && state.preferredTime)
  ) {
    issues.push("The bot made an availability claim before enough schedule details were supplied.");
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
      issues.push(`The bot asked for ${type} even though it was already supplied.`);
    }
  }

  if (countQuestions(reply) > 1) {
    issues.push("The reply contains more than one workflow question.");
  }

  if (step.expectAnswerAny && !hasAny(reply, step.expectAnswerAny)) {
    issues.push(`Expected an informational answer containing one of: ${step.expectAnswerAny.join(", ")}.`);
  }
  if (step.expectWorkflowContinuation && countQuestions(reply) !== 1) {
    issues.push("Expected the side-question answer to continue with exactly one workflow question.");
  }
  if (step.forbidAnswerAny && hasAny(reply, step.forbidAnswerAny)) {
    issues.push(`Reply contained forbidden stale content: ${step.forbidAnswerAny.join(", ")}.`);
  }
  for (const type of step.expectQuestionTypes || []) {
    if (!asksQuestionType(reply, type)) {
      issues.push(`Expected the bot to ask for ${type}.`);
    }
  }
  for (const type of step.forbidQuestionTypes || []) {
    if (asksQuestionType(reply, type)) {
      issues.push(`The bot repeated a question for already supplied ${type}.`);
    }
  }

  if (!reply.trim()) {
    issues.push("The bot returned an empty reply.");
  }
  if (/^[\s\p{P}]+$/u.test(reply)) {
    issues.push("The bot returned punctuation-only content.");
  }
  if (reply.trim().toLowerCase() === "thanks.") {
    issues.push('The bot returned the meaningless fallback "Thanks.".');
  }

  return [...new Set(issues)];
}

async function takeScreenshot(page, scenarioSlug, stepNumber, phase) {
  const filename = `${scenarioSlug}-${String(stepNumber).padStart(2, "0")}-${phase}.png`;
  const fullPath = path.join(SCREENSHOT_DIR, filename);
  await page.screenshot({ path: fullPath, fullPage: true });
  return path.relative(ROOT, fullPath).replace(/\\/g, "/");
}

function writeAnnotation(scenarioSlug, stepNumber, issues, screenshot) {
  const filename = `${scenarioSlug}-${String(stepNumber).padStart(2, "0")}-issues.txt`;
  const fullPath = path.join(SCREENSHOT_DIR, filename);
  const body = [
    `Screenshot: ${screenshot}`,
    "",
    "Detected issues:",
    ...issues.map((issue) => `- ${issue}`)
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

      updateStateFromUser(state, step.text);
      await page.locator(SELECTORS.input).fill(step.text);
      await page.locator(SELECTORS.sendButton).click();
      await page.locator(SELECTORS.userMessages).last().waitFor({ state: "visible" });
      const userScreenshot = await takeScreenshot(page, scenarioSlug, stepNumber, "user");

      const botReply = await waitForBotReply(page, previousBotCount);
      updateStateFromBot(state, botReply);
      const botScreenshot = await takeScreenshot(page, scenarioSlug, stepNumber, "bot");
      const issues = detectIssues({ state, reply: botReply, step });
      let annotationFile = null;

      if (issues.length) {
        annotationFile = writeAnnotation(scenarioSlug, stepNumber, issues, botScreenshot);
        result.detectedIssues.push(
          ...issues.map((issue) => ({ step: stepNumber, message: issue }))
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
      finalIssues.push(
        `Final reply did not request one of: ${scenario.finalExpectQuestionAny.join(", ")}.`
      );
    }
    if (scenario.finalForbidAnswerAny && hasAny(finalReply, scenario.finalForbidAnswerAny)) {
      finalIssues.push(
        `Final reply contained forbidden wording: ${scenario.finalForbidAnswerAny.join(", ")}.`
      );
    }
    if (finalIssues.length && result.conversation.length) {
      const finalTurn = result.conversation.at(-1);
      finalTurn.detectedIssues.push(...finalIssues);
      result.detectedIssues.push(
        ...finalIssues.map((message) => ({
          step: scenario.messages.length,
          message
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
      message: `Runner error: ${error.message}`
    });
    const failureScreenshot = await takeScreenshot(
      page,
      scenarioSlug,
      result.conversation.length + 1,
      "runner-error"
    ).catch(() => null);
    if (failureScreenshot) {
      writeAnnotation(
        scenarioSlug,
        result.conversation.length + 1,
        [error.stack || error.message],
        failureScreenshot
      );
    }
  } finally {
    result.finishedAt = new Date().toISOString();
    result.status = result.detectedIssues.length ? "failed" : "passed";
    await context.close();
  }

  return result;
}

function markdownReport(report) {
  const lines = [
    "# Veronika Wellness Chatbot QA Report",
    "",
    `- Run started: ${report.startedAt}`,
    `- Base URL: ${report.baseUrl}`,
    `- Timezone: Europe/London`,
    `- Result: **${report.summary.failed ? "FAILED" : "PASSED"}**`,
    `- Scenarios: ${report.summary.total}`,
    `- Passed: ${report.summary.passed}`,
    `- Failed: ${report.summary.failed}`,
    ""
  ];

  for (const scenario of report.scenarios) {
    lines.push(`## ${scenario.testName}`);
    lines.push("");
    lines.push(`Status: **${scenario.status.toUpperCase()}**`);
    lines.push("");

    for (const turn of scenario.conversation) {
      lines.push(`### Step ${turn.step}`);
      lines.push("");
      lines.push(`**Customer:** ${turn.userMessage}`);
      lines.push("");
      lines.push(`**Assistant:** ${turn.botReply}`);
      lines.push("");
      lines.push(`- User screenshot: \`${turn.screenshots.afterUserMessage}\``);
      lines.push(`- Bot screenshot: \`${turn.screenshots.afterBotReply}\``);
      if (turn.annotationFile) {
        lines.push(`- Annotation: \`${turn.annotationFile}\``);
      }
      for (const issue of turn.detectedIssues) {
        lines.push(`- Issue: ${issue}`);
      }
      lines.push("");
    }

    if (!scenario.detectedIssues.length) {
      lines.push("No issues detected.");
      lines.push("");
    }
  }

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
  fs.writeFileSync(MARKDOWN_REPORT, markdownReport(report), "utf8");

  console.log(`JSON report: ${JSON_REPORT}`);
  console.log(`Markdown report: ${MARKDOWN_REPORT}`);
  console.log(`Passed: ${report.summary.passed}; Failed: ${report.summary.failed}`);
  process.exitCode = report.summary.failed ? 1 : 0;
}

main().catch((error) => {
  ensureDirectories();
  const failure = {
    startedAt: new Date().toISOString(),
    baseUrl: BASE_URL,
    summary: { total: 0, passed: 0, failed: 1 },
    runnerError: error.stack || error.message,
    scenarios: []
  };
  fs.writeFileSync(JSON_REPORT, `${JSON.stringify(failure, null, 2)}\n`, "utf8");
  fs.writeFileSync(
    MARKDOWN_REPORT,
    `# Veronika Wellness Chatbot QA Report\n\nRunner failed:\n\n\`\`\`\n${failure.runnerError}\n\`\`\`\n`,
    "utf8"
  );
  console.error(error);
  process.exitCode = 1;
});
