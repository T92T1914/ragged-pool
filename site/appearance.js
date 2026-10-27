/* Resolve only appearance before paint; the explanation works without scripts. */
(function () {
  'use strict';
  const key = 'ragged-pool.appearance.v1';
  const root = document.documentElement;
  let choice = 'obscur';
  try {
    const saved = localStorage.getItem(key);
    if (['auto', 'clair', 'obscur'].includes(saved)) choice = saved;
  } catch (_) { /* The unsaved Obscur default remains usable. */ }
  root.dataset.appearance = choice;
  document.addEventListener('DOMContentLoaded', () => {
    const control = document.getElementById('appearance');
    if (!control) return;
    control.value = choice;
    control.disabled = false;
    control.addEventListener('change', () => {
      choice = ['auto', 'clair', 'obscur'].includes(control.value) ? control.value : 'obscur';
      root.dataset.appearance = choice;
      try { localStorage.setItem(key, choice); }
      catch (_) { /* The explicit in-memory choice still works. */ }
    });
  });
})();

