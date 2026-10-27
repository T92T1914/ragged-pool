# Static project explanation

This page explains the accepted CPU component without installing or executing it.
Its ordinary text and diagram work without JavaScript. The optional script changes
only appearance: Obscur is the first-visit and no-script default, while an explicit
Auto, Clair or Obscur choice stays in this project's browser storage.

Build with Python 3.11's standard library from the repository:

```sh
python tools/test_site.py
python tools/build_site.py
python tools/build_site.py --check-built
```

The builder publishes exactly the three maintained site files and two existing
Obscur PNGs, plus a source/asset record. It does not copy the package, environment,
private data or arbitrary repository contents. It imports no PyTorch/native code.
The record distinguishes the page revision and dirty source state from the
accepted v0.2.0 runtime/release revision. This is presentation work, with no new
numerical, device, gradient, performance or installed-consumer qualification.

The family roles reuse the reviewed
[theme revision](https://github.com/T92T1914/clair-obscur-themes/blob/7a57fe750ff50205a17e1d342106a0d3f2777159/tokens.json).
Local six-face Inter has system-font fallbacks; no font binary or remote font
download is distributed. Original diagram bytes and their generator remain in
docs. Dark diagrams are intentional even when a visitor chooses a Clair page.

The normal Project explainer workflow validates and uploads only this static
payload. Deployment requires GitHub Pages to be configured for GitHub Actions in
the existing public repository. Successful local source/build checks do not prove
that setting exists or that the published URL has been accepted.

