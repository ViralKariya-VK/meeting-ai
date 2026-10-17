"""Deterministic synthetic seed data for the classical sentence classifier.

IMPORTANT: this is *synthetic, template-generated* data, used only to bootstrap the demo.
Held-out numbers on it are optimistic. The separate hand-written test set
(models/data/hand_labelled_test.csv) is never used for training and gives a more honest estimate.
Replace/augment with real meeting sentences (see scripts/silver_label_with_groq.py) for serious use.
"""
from __future__ import annotations

import random

NAMES = ["Rahul", "Ananya", "Priya", "Mark", "Sarah", "John", "Meera", "David", "Elena", "Omar", "Chen", "Lisa",
         "Tom", "Aisha", "Kevin", "Nina", "Sam", "Julia", "Arjun", "Fatima", "Peter", "Grace", "Leo", "Zoe"]
TASKS = ["send the project report", "update the dashboard", "review the design doc", "book the venue",
         "fix the login bug", "prepare the slides", "share the budget sheet", "schedule the client call",
         "draft the release notes", "complete the charts", "test the payment flow", "write the user guide",
         "contact the vendor", "migrate the database", "finalise the contract", "update the roadmap",
         "collect the survey responses", "set up the staging server", "email the customers", "clean up the backlog"]
TASK_ING = ["sending the project report", "updating the dashboard", "reviewing the design doc", "booking the venue",
            "fixing the login bug", "preparing the slides", "the budget sheet", "the client call", "the release notes",
            "the payment testing", "the vendor relationship", "the database migration"]
DEADLINES = ["by Friday", "by Monday", "tomorrow", "by end of day", "next Tuesday", "by the 15th",
             "before the next meeting", "by Wednesday", "this week", "by 12 October", "by the end of the month",
             "by Thursday afternoon", "next week", "by tomorrow morning", "on Monday"]
DECISIONS = ["launch the beta on 1 November", "move the release to next month", "use PostgreSQL for the backend",
             "drop the legacy export feature", "hire two more engineers", "postpone the redesign",
             "adopt weekly sprints", "go with the vendor's second quote", "keep the current pricing",
             "cancel the Friday standup", "ship the smaller feature set first", "stay with the current vendor",
             "extend the trial to thirty days", "freeze the scope until launch"]
OPTIONS = ["option B", "the blue design", "the cloud-hosted setup", "the smaller scope", "Plan A", "the first proposal",
           "the monthly plan", "the new logo"]
THINGS = ["design", "release notes", "budget", "migration", "contract", "dashboard", "payment flow", "test results",
          "marketing plan", "client proposal", "roadmap", "onboarding flow", "vendor quote", "launch date"]
OBS = ["looks fine", "needs more work", "went well", "was delayed", "is still being discussed",
       "has a few open issues", "got good feedback", "is behind schedule", "is on track", "is more expensive than expected"]
FILLERS = ["", "", "", "Okay, ", "So ", "Alright, ", "Right, ", "Well, ", "Yeah, "]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _action(r: random.Random) -> str:
    n, t, d = r.choice(NAMES), r.choice(TASKS), r.choice(DEADLINES)
    ing, n2 = r.choice(TASK_ING), r.choice(NAMES)
    forms = [
        f"{n} will {t} {d}.", f"{n} is going to {t} {d}.", f"I'll {t} {d}.", f"I will {t} {d}.",
        f"Please {t} {d}.", f"{n}, please {t} {d}.", f"{n} is responsible for {ing}.",
        f"Action item: {n} to {t} {d}.", f"We need {n} to {t} {d}.", f"{n} agreed to {t} {d}.",
        f"Let's have {n} {t} {d}.", f"I'll take this one and {t}.", f"{n} will take care of {ing}.",
        f"Could you please {t} {d}?", f"{n} has to {t} {d}.", f"{n} and {n2} will {t} {d}.",
        f"Next step: {n} will {t}.", f"I'll follow up and {t} {d}.", f"{n} said she'll {t} {d}.",
        f"{n} promised to {t} {d}.", f"Can you {t} {d}, {n}?",
    ]
    return r.choice(FILLERS) + _cap(r.choice(forms))


def _decision(r: random.Random) -> str:
    x, o, th = r.choice(DECISIONS), r.choice(OPTIONS), r.choice(THINGS)
    forms = [
        f"We have decided to {x}.", f"We agreed to {x}.", f"The team approved the {th}.", f"It's settled: we will {x}.",
        f"Final decision: {x}.", f"Let's go with {o}.", f"We will go ahead with {o}.", f"The decision is to {x}.",
        f"Everyone agreed that we should {x}.", f"Approved: the {th}.", f"We're going with {o}.",
        f"We settled on {o}.", f"So the decision is {o}.", f"The board approved the {th}.",
        f"We voted to {x}.", f"It's final, we {x}.", f"We won't {x} this year, that's decided.",
    ]
    return r.choice(FILLERS) + _cap(r.choice(forms))


def _discussion(r: random.Random) -> str:
    n, th, o, t = r.choice(NAMES), r.choice(THINGS), r.choice(OBS), r.choice(TASKS)
    forms = [
        f"The {th} {o}.", f"I think the {th} {o}.", f"{n} mentioned that the {th} {o}.",
        f"Last week the {th} {o}.", f"I'm not sure the {th} is realistic.", f"We should probably look at the {th} at some point.",
        f"It might make sense to revisit the {th} sometime.", f"I will not be able to {t}.",
        f"{n} did not {t} last week.", f"{n} didn't get a chance to look at the {th}.",
        f"Customers keep asking about the {th}.", "Thanks everyone for joining.", "Let me share my screen.",
        "Okay, moving on to the next topic.", "That approach worked well in the pilot.",
        f"Honestly the {th} was a mess but we learned a lot.", f"Maybe we can try something different with the {th} later.",
        f"The numbers for the {th} dropped a bit this month.", f"I'm worried the {th} is a bit optimistic.",
        f"Someone mentioned the {th} might slip.", f"We talked about the {th} last time too.",
        f"{n} thinks we could automate the {th} eventually.", f"I won't be around on Friday.",
        f"I finished the {th} yesterday.", f"{n} finished the {th} last week.", f"We completed the {th} on Monday.",
        f"The {th} is done now.", f"The {th} reached {r.randint(40, 95)} percent.",
        f"Spending on the {th} went up by {r.randint(3, 30)} percent this quarter.",
        f"We have about {r.randint(10, 400)} open items on the {th}.", "Good morning everyone, thanks for joining.",
        "Welcome everyone, let's get started.", "Okay, moving on to the next item.", f"Quick standup on the {th}.",
        "That's all for today, thanks.", f"One more thing about the {th}.", f"Now the big question about the {th}.",
        f"I think the {th} is more convincing than the slides.", "Honestly it could slip.",
        f"The first quote for the {th} seemed high.", f"It confuses refusals with commitments sometimes.",
        f"Users like the {th} but a few people found it confusing.", f"The beta feedback on the {th} has been mostly positive.",
        f"Nobody has looked at the {th} since the audit.", f"The {th} baseline reached {r.randint(60, 90)} percent accuracy.",
        f"{n} said the {th} was faster than expected.", "Great, that makes sense.",
    ]
    return r.choice(FILLERS) + _cap(r.choice(forms))


def _question(r: random.Random) -> str:
    n, th = r.choice(NAMES), r.choice(THINGS)
    forms = [
        f"Can we review the {th}?", f"What is the status of the {th}?", f"Who is handling the {th}?",
        f"When is the deadline for the {th}?", f"Do we have budget for the {th}?", f"Why did the {th} fail?",
        f"How long will the {th} take?", f"Is anyone blocked on the {th}?", f"Are we sure about the {th}?",
        f"Did {n} finish the {th}?", f"What did we agree on for the {th}?", f"Could we go through the {th} again?",
        f"Does anyone know who owns the {th}?", f"Is {n} still working on the {th}?", f"Where can I find the {th}?",
        f"Shall we start with the {th}?", f"Have we sent the {th} yet?", f"Should we also update the {th}?",
        f"Do we launch on the first or push the {th} to next month?", f"Does anyone know when the {th} is due?",
        f"How does it handle negation in the {th}?", f"Anything else on the {th}?",
    ]
    return r.choice(FILLERS) + _cap(r.choice(forms))


def generate(n_per_class: int = 260, seed: int = 7) -> list[tuple[str, str]]:
    r = random.Random(seed)
    out: dict[str, str] = {}
    makers = {"ACTION_ITEM": _action, "DECISION": _decision, "DISCUSSION": _discussion, "QUESTION": _question}
    for label, fn in makers.items():
        count, tries = 0, 0
        while count < n_per_class and tries < n_per_class * 40:
            tries += 1
            s = fn(r).strip()
            if s not in out:
                out[s] = label
                count += 1
    return list(out.items())
