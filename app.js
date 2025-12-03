// ====== CONFIG ======
const API_URL = "http://127.0.0.1:5000/api/analyze"; // change this to your actual backend endpoint

// ====== DOM ELEMENTS ======
const queryInput = document.getElementById("queryInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const statusMessage = document.getElementById("statusMessage");

const categoryPill = document.getElementById("categoryPill");
const subCategoryPill = document.getElementById("subCategoryPill");
const categoryNote = document.getElementById("categoryNote");

const lawsList = document.getElementById("lawsList");
const lawsNote = document.getElementById("lawsNote");

const adviceText = document.getElementById("adviceText");

const casesList = document.getElementById("casesList");
const casesNote = document.getElementById("casesNote");

// ====== HELPERS ======

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  if (isLoading) {
    statusMessage.textContent = "Analyzing your query…";
  } else {
    // don't overwrite error/success messages here
  }
}

function clearResults() {
  categoryPill.textContent = "Category: –";
  subCategoryPill.textContent = "Sub-category: –";
  categoryNote.textContent = "";
  lawsList.innerHTML = "";
  lawsNote.textContent = "";
  adviceText.textContent = "No advice yet. Submit a query to see guidance.";
  casesList.innerHTML = "";
  casesNote.textContent = "";
  statusMessage.textContent = "";
}

function renderCategory(category, subtopicId, subtopicTitle) {
  categoryPill.textContent = category ? `Category: ${category}` : "Category: –";

  if (subtopicId) {
    const label = subtopicTitle ? subtopicTitle : subtopicId;
    subCategoryPill.textContent = `Sub-category: ${label}`;
  } else {
    subCategoryPill.textContent = "Sub-category: –";
  }

  if (category && !["TENANCY", "LABOUR"].includes(category)) {
    categoryNote.textContent =
      "Detailed sub-category and advice are currently available only for Tenancy and Labour issues.";
  } else {
    categoryNote.textContent = "";
  }
}

function renderLaws(lawsArr) {
  lawsList.innerHTML = "";
  if (!lawsArr || lawsArr.length === 0) {
    lawsNote.textContent = "No specific laws could be identified for this query.";
    return;
  }
  lawsArr.forEach((law) => {
    const li = document.createElement("li");
    li.textContent = law;
    lawsList.appendChild(li);
  });
  lawsNote.textContent =
    "These provisions are suggested based on similarity and may not cover all applicable laws.";
}

function renderAdvice(advice) {
  if (!advice) {
    adviceText.textContent =
      "Advice is not available for this query. Currently detailed guidance is provided only for Tenancy and Labour.";
  } else {
    adviceText.textContent = advice;
  }
}

function renderCases(casesArr) {
  casesList.innerHTML = "";
  if (!casesArr || casesArr.length === 0) {
    casesNote.textContent =
      "No specific reference cases were found for this query yet.";
    return;
  }

  casesArr.forEach((c) => {
    const li = document.createElement("li");

    const title = c.title || "Case";
    const court = c.court ? `, ${c.court}` : "";
    const year = c.year ? `, ${c.year}` : "";
    const citation = c.citation ? ` (${c.citation})` : "";

    const heading = document.createElement("div");
    heading.innerHTML = `<strong>${title}</strong>${court}${year}${citation}`;

    li.appendChild(heading);

    if (c.summary) {
      const summary = document.createElement("div");
      summary.textContent = c.summary;
      summary.classList.add("small-note");
      li.appendChild(summary);
    }

    if (c.link) {
      const link = document.createElement("a");
      link.href = c.link;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "View judgment";
      link.classList.add("small-note");
      li.appendChild(link);
    }

    casesList.appendChild(li);
  });

  casesNote.textContent =
    "Cases are suggested based on similarity and are for reference only.";
}

// ====== MAIN REQUEST ======

async function analyzeQuery() {
  const query = queryInput.value.trim();

  if (!query) {
    statusMessage.textContent = "Please describe your legal issue before submitting.";
    return;
  }

  clearResults();
  setLoading(true);

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!res.ok) {
      throw new Error(`Server returned status ${res.status}`);
    }

    const data = await res.json();

    // Expected fields (adjust names to match your backend)
    const category = data.category || null;
    const subtopicId = data.subtopic_id || null;
    const subtopicTitle = data.subtopic_title || null;
    const laws = data.laws || [];
    const advice = data.advice || "";
    const referenceCases = data.reference_cases || [];

    renderCategory(category, subtopicId, subtopicTitle);
    renderLaws(laws);
    renderAdvice(advice);
    renderCases(referenceCases);

    statusMessage.textContent = "Analysis completed.";
  } catch (err) {
    console.error(err);
    statusMessage.textContent =
      "Something went wrong while contacting the server. Please try again later.";
  } finally {
    setLoading(false);
  }
}

// ====== EVENT LISTENERS ======

analyzeBtn.addEventListener("click", analyzeQuery);

clearBtn.addEventListener("click", () => {
  queryInput.value = "";
  clearResults();
});

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    analyzeQuery();
  }
});
