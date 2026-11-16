# Accessibility

AXIOM is a terminal-first local AI tool with a static landing page, and its user-facing surfaces should be usable by as many people as practical, including people who navigate by keyboard, use assistive technology, zoom or magnify content, or need clear and predictable language. This document describes our accessibility priorities, contributor expectations, and how to report barriers. It is a statement of intent and current practice, not a claim of verified conformance to an accessibility standard.

## Priorities

We aim to make the AXIOM landing page, terminal guidance, and documentation understandable and operable across different access needs. Priorities include:

- **Keyboard access:** interactive controls should be reachable and usable without a mouse, with a visible focus indicator and a logical focus order.
- **Structure and names:** use meaningful page headings, semantic elements, programmatic labels, and descriptive names for controls and icons.
- **Readable presentation:** use legible text, sufficient contrast, responsive layouts, and avoid relying on colour alone to communicate status.
- **Clear feedback:** validation errors and operation status should be described in text and associated with the relevant controls where practical.
- **Predictable interaction:** avoid unexpected context changes, preserve entered data where possible, and respect reduced-motion preferences when animations are present.
- **Documentation:** use plain language, define specialist terms when necessary, and provide meaningful link text.

These are goals for ongoing development. AXIOM has not yet completed a comprehensive accessibility evaluation, so no WCAG conformance level is claimed.

## Contributor expectations

Accessibility is part of feature quality. For user-facing changes, contributors should:

1. Prefer native semantic HTML controls over custom interactive elements when possible.
2. Ensure forms have visible labels, errors are understandable, and icon-only controls have accessible names.
3. Check keyboard navigation, focus visibility, zoomed layouts, and relevant loading, empty, success, and error states.
4. Avoid conveying meaning through colour, shape, or animation alone; consider reduced-motion preferences.
5. Update user documentation when a change affects instructions or interaction.
6. Include the checks performed in the pull request description. Screenshots can help explain visual issues, but do not replace keyboard or assistive-technology checks.

Run the repository's available validation before submitting changes, including `npm run lint`, `npm run build`, and the relevant Playwright tests when applicable. Passing these checks does not by itself establish accessibility conformance; dedicated manual checks are still needed.

## Reporting accessibility issues

Report an accessibility barrier through [AXIOM-AI GitHub Issues](https://github.com/manit6752025/AXIOM-AI/issues) with a title that starts with **Accessibility:**. Please describe:

- the page, URL, or feature affected;
- the task you were trying to complete and what happened;
- the browser and operating system, if known;
- the input method or assistive technology, if relevant;
- a workaround, if you found one.

Screenshots or recordings are optional. You do not need to disclose a diagnosis or other personal health information. Do not include passwords, tokens, private model data, or other sensitive information in a public issue. If the report also exposes a security vulnerability, use the repository's private security reporting process instead.

### Severity

Maintainers can assign or adjust priority during triage; reporters do not need to choose a severity.

- **Blocker:** a user cannot complete a core task and there is no practical workaround, such as being unable to complete first-time setup or sign in using the available controls.
- **High:** an important task is very difficult or inaccessible for a group of users, and the workaround is substantially limiting.
- **Medium:** a barrier affects a secondary task or has a workable but inconvenient alternative.
- **Low:** a minor usability or clarity issue that does not prevent task completion.

These labels guide discussion and are not a promise of a fixed resolution deadline.

### How we respond

The project maintainers will review reports as capacity permits, ask for clarification when needed, and update the issue when status or a workaround is known. Response and fix timing depends on severity, impact, available maintainers, and technical complexity; we do not currently promise a fixed acknowledgement or resolution time. Where practical, we will invite the reporter to verify a proposed fix before the issue is closed.

## Ownership and maintenance

The AXIOM maintainers under NetCore Technologies are responsible for reviewing accessibility reports, considering accessibility in user-facing changes, and keeping this statement current. We will revisit it when major interface changes ship and during release preparation. If responsibility transfers, the repository maintainers should update this section and the relevant contribution guidance.

## Supported environments

The current end-to-end suite exercises the static landing page in Playwright's Chromium browser. This is useful regression coverage, but it is not a complete compatibility or accessibility audit. Other browsers, operating systems, mobile form factors, terminal emulators, and specific screen-reader/browser combinations have not been systematically certified by this statement. Please include the environment details when reporting a problem.

## Known limitations

A comprehensive manual audit across keyboard-only use, screen readers, magnification, high-contrast settings, and reduced-motion preferences has not yet been completed. Automated lint, build, and end-to-end tests may not detect all accessibility barriers. Until that audit is complete, treat accessibility as actively being improved rather than assuming every flow is barrier-free. Known issues should be tracked in GitHub Issues and linked here as they are confirmed.

## Feedback and improvements

Suggestions for improving this statement or AXIOM's accessibility practices are welcome through [GitHub Issues](https://github.com/manit6752025/AXIOM-AI/issues) using the **Accessibility:** prefix. Please report an active barrier using the process above so it can be triaged with the affected task and context.
