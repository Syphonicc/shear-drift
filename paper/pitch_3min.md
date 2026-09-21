# Shear Drift — 3-minute pitch (spoken, ~430 words)

Timings are for a slow, clear delivery. Stage directions in brackets.

---

**[0:00 — the problem, 45 s]**

About one adult in thirty is walking around with a brain aneurysm: a small balloon on the wall of an artery. Most of them will never burst. A few will, and when one does, roughly a third of those people die and another third are left disabled.

So the doctor's job is a bet. Treat it — with surgery or a coil through the groin — and you accept a real risk of stroke from the procedure itself. Leave it — and you accept a small risk, every year, of the balloon bursting. Today that bet is made mostly on size and shape from a scan. Size is a blunt instrument: small aneurysms rupture too, and large ones often don't.

**[0:45 — how they're trying to solve it, 45 s]**

What actually decides whether the wall weakens is how blood *rubs* against it, heartbeat after heartbeat. In some spots the blood pushes forward in one part of the beat and slides back in the next. Where that back-and-forth happens, blood lingers, the wall gets inflamed, and that is where aneurysms grow and burst.

We can compute that rubbing for an individual patient. It's called a flow simulation. The catch: one patient takes a specialist and hours to days of computer time. Hospitals see thousands of these scans. So it stays in research papers, not in clinics.

**[1:30 — the new hope and its blind spot, 45 s]**

The obvious fix, and the one everyone is building right now, is AI. Train a model on thousands of simulations, then predict the flow for a new patient in seconds. The papers report ninety-seven, ninety-eight percent accuracy. Sounds solved.

Here's the blind spot. Think of the blood as taking ten steps forward and nine steps back. The number doctors care about — how long blood *lingers* — depends on where you end up: one step. Overall accuracy measures the nineteen steps taken. If the AI miscounts one step, overall accuracy drops five percent. The lingering number is off by a hundred percent. Same tiny mistake, opposite consequences. And it happens exactly where the aneurysm is.

**[2:15 — what we built, 45 s]**

We built the check that catches this. Give it any AI flow model and it does three things. It shows, on the actual artery, where the lingering number can and can't be trusted. It measures *how much* the trust drops, with a formula we fitted on hundreds of exact cases and confirmed on real patient simulations and on a real AI model we trained. And — this is the part I'm proudest of — from three numbers every AI paper already reports, it tells you *what kind* of mistake the model is making, and therefore whether to re-tune it, smooth it, or go back and retrain. We tested that last part on a published model from Imperial and Singapore, without touching their code. It called it correctly.

**[2:55 — close, 10 s]**

It's a unit test for medical AI. Accuracy dashboards say fine. This says *where* fine isn't good enough.

---

## Notes for delivery
- "One in thirty" and "a third die, a third disabled" are the standard order-of-magnitude figures for unruptured intracranial aneurysm prevalence and subarachnoid haemorrhage outcomes; say "roughly" and don't get drawn into decimals.
- If asked "which three numbers": TAWSS, OSI and RRT — "average rubbing, how much it reverses, and how long blood lingers."
- If asked "what did the Imperial model get wrong": nothing wrong — its fingerprint says its error is a persistent spatial bias, not a timing error, so the fix is a better model, not re-timing. That's a diagnosis, not a takedown.
- The AAA (belly aorta) case is the reversal example if someone asks why the brain case looks so green: brain arteries barely reverse; that's why it's the control.
