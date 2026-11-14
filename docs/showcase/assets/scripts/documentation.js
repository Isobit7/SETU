/**
 * SETU Documentation JavaScript Engine (Offline-First)
 * Handles Theme Toggling, Code Copying, Offline Search, and Sidebar Navigation.
 */

(function () {
  'use strict';

  // 1. Theme Toggle System
  const THEME_KEY = 'setu_docs_theme';

  function initTheme() {
    const savedTheme = localStorage.getItem(THEME_KEY);
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const activeTheme = savedTheme || (prefersDark ? 'dark' : 'light');

    document.documentElement.setAttribute('data-theme', activeTheme);
    updateThemeButtonText(activeTheme);
  }

  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', nextTheme);
    localStorage.setItem(THEME_KEY, nextTheme);
    updateThemeButtonText(nextTheme);
  }

  function updateThemeButtonText(theme) {
    const btn = document.getElementById('theme-toggle-btn');
    if (!btn) return;
    if (theme === 'dark') {
      btn.innerHTML = '☀️ Light Mode';
    } else {
      btn.innerHTML = '🌙 Dark Mode';
    }
  }

  // 2. Code Snippet Copy-to-Clipboard
  function initCodeCopy() {
    const codeBlocks = document.querySelectorAll('.code-container');
    codeBlocks.forEach((block) => {
      const copyBtn = block.querySelector('.copy-btn');
      const pre = block.querySelector('pre');
      if (copyBtn && pre) {
        copyBtn.addEventListener('click', () => {
          const codeText = pre.innerText;
          navigator.clipboard.writeText(codeText).then(() => {
            const originalText = copyBtn.innerText;
            copyBtn.innerText = 'Copied!';
            copyBtn.style.background = '#059669';
            setTimeout(() => {
              copyBtn.innerText = originalText;
              copyBtn.style.background = '';
            }, 2000);
          }).catch(() => {
            copyBtn.innerText = 'Error';
          });
        });
      }
    });
  }

  // 3. Client-Side Offline Search Filter
  function initSearch() {
    const searchInput = document.getElementById('doc-search-input');
    if (!searchInput) return;

    searchInput.addEventListener('input', (e) => {
      const query = (e.target.value || '').toLowerCase().trim();
      const searchableElements = document.querySelectorAll('h2, h3, .card, tr');

      if (!query) {
        searchableElements.forEach((el) => {
          if (el.dataset.searchHidden) {
            el.style.display = '';
            delete el.dataset.searchHidden;
          }
        });
        return;
      }

      searchableElements.forEach((el) => {
        const text = el.innerText.toLowerCase();
        if (text.includes(query)) {
          el.style.display = '';
          delete el.dataset.searchHidden;
        } else {
          // Only hide cards or table rows, keep headers visible for structure
          if (el.classList.contains('card') || el.tagName === 'TR') {
            el.style.display = 'none';
            el.dataset.searchHidden = 'true';
          }
        }
      });
    });
  }

  // 4. Highlight Active Navigation Link
  function initActiveNav() {
    const currentPath = window.location.pathname.split('/').pop() || 'index.html';
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach((link) => {
      const href = link.getAttribute('href');
      if (href && (href === currentPath || (currentPath === '' && href === 'index.html'))) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });
  }

  // Initialization on DOM Ready
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initCodeCopy();
    initSearch();
    initActiveNav();

    const toggleBtn = document.getElementById('theme-toggle-btn');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', toggleTheme);
    }
  });
})();

