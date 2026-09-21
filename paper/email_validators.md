Subject: 10-minute test of a "unit test" for hemodynamic ML surrogates — would you try it?

Dear Prof. <NAME>,

I am a student working on error diagnostics for deep-learning surrogates of time-resolved wall shear stress. Your work on <THEIR PAPER / TOPIC> is one of the reasons I think this matters, so I would value your opinion.

The tool is a single web page (link below, no install, private link, ~3 MB). It takes a surrogate's error on the WSS field and shows, on a patient cerebral aneurysm and an AAA, where the residence-time biomarker (RRT) can no longer be trusted, and whether the fix is re-timing the model's clock or improving the model. The underlying observation is that TAWSS, OSI and RRT errors are coupled through the cancellation ratio |∫τ dt| / ∫|τ| dt, so the three numbers most surrogate papers already report tell you what kind of error the model has. A short technical note is attached.

Would you be willing to spend ~10 minutes on it and answer four questions?
1. Does the RRT-vs-TAWSS framing match what you see in your own surrogate/CFD comparisons?
2. Is there a case in your experience where the diagnostic would give the wrong advice?
3. Which of the three biomarkers do you actually report to clinicians, and would this change how you validate it?
4. May I quote your answer (attributed, or anonymised as "<field>, <institution type>") on the project page?

Link: <EXPLORER URL>
Note: attached / <ZENODO DOI>
Code: github.com/Syphonicc/shear-drift (public from <DATE>)

This is being submitted to the UnivaXBio hackathon on 6 October, which is why the ask is short-notice; I would be grateful for any reply, including "no".

Best regards,
Suvam Samanta
<affiliation>
