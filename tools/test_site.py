"""Negative admission checks for the static page, with no native/PyTorch import."""
import unittest

from build_site import FILES, ROOT, validate_page, validate_sources


class StaticPageTests(unittest.TestCase):
    def setUp(self):
        self.page = (ROOT / "site/index.html").read_text(encoding="utf-8")
        self.names = set(FILES.values()) | {"site-record.json"}

    def test_actual_sources_keep_retained_diagrams_and_local_routes(self):
        self.assertEqual(set(validate_sources()), set(FILES.values()))

    def test_missing_image_alternative_is_refused(self):
        changed = self.page.replace('alt="Conceptual CPU example:', 'data-description="Conceptual CPU example:')
        with self.assertRaisesRegex(ValueError, "Missing diagram alternative"):
            validate_page(changed, self.names)

    def test_missing_local_image_and_anchor_are_refused(self):
        for old, new, reason in [
            ('srcset="ragged-pool-obscur-narrow.png"', 'srcset="missing.png"', "Unpublished local"),
            ('href="#worked-example"', 'href="#missing-example"', "Missing page anchor"),
        ]:
            with self.subTest(reference=new):
                with self.assertRaisesRegex(ValueError, reason):
                    validate_page(self.page.replace(old, new), self.names)

    def test_late_preference_and_missing_no_script_default_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Missing no-script"):
            validate_page(self.page.replace('data-appearance="obscur"', 'data-appearance="auto"'), self.names)
        changed = self.page.replace('<script src="appearance.js"></script>', '')
        changed = changed.replace('</head>', '<script src="appearance.js"></script></head>')
        with self.assertRaisesRegex(ValueError, "before styles"):
            validate_page(changed, self.names)


if __name__ == "__main__":
    unittest.main()

