/* KuaiSearch Website JavaScript */

// ============ PARTICLE ANIMATION ============
function createParticles() {
  const container = document.getElementById('particles');
  if (!container) return;

  const colors = ['rgba(74,144,226,0.4)', 'rgba(124,58,237,0.3)', 'rgba(16,185,129,0.3)'];

  for (let i = 0; i < 30; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    const size = Math.random() * 5 + 2;
    particle.style.cssText = `
      width: ${size}px;
      height: ${size}px;
      left: ${Math.random() * 100}%;
      background: ${colors[Math.floor(Math.random() * colors.length)]};
      animation-duration: ${Math.random() * 15 + 10}s;
      animation-delay: ${Math.random() * 10}s;
    `;
    container.appendChild(particle);
  }
}

// ============ NAVBAR SCROLL EFFECT ============
function initNavbar() {
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 80) {
      navbar.style.background = 'rgba(15, 17, 23, 0.97)';
    } else {
      navbar.style.background = 'rgba(15, 17, 23, 0.85)';
    }
  });
}

// ============ BENCHMARK TABS ============
function initBenchmarkTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;

      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetContent = document.getElementById(`tab-${target}`);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });
}

// ============ DATA PREVIEW TABS ============
function initPreviewTabs() {
  const previewBtns = document.querySelectorAll('.preview-tab-btn');
  const previewContents = document.querySelectorAll('.preview-content');

  previewBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.preview;

      previewBtns.forEach(b => b.classList.remove('active'));
      previewContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetContent = document.getElementById(`preview-${target}`);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });
}

// ============ BIBTEX COPY ============
function initBibtexCopy() {
  const copyBtn = document.getElementById('copyBibtex');
  const bibtexCode = document.getElementById('bibtex-code');

  if (!copyBtn || !bibtexCode) return;

  copyBtn.addEventListener('click', async () => {
    const text = bibtexCode.innerText || bibtexCode.textContent;
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.classList.add('copied');
      copyBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
        Copied!
      `;
      setTimeout(() => {
        copyBtn.classList.remove('copied');
        copyBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
          </svg>
          Copy
        `;
      }, 2500);
    } catch (err) {
      // Fallback
      const el = document.createElement('textarea');
      el.value = text;
      document.body.appendChild(el);
      el.select();
      document.execCommand('copy');
      document.body.removeChild(el);
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.textContent = 'Copy'; }, 2000);
    }
  });
}

// ============ SCROLL ANIMATIONS ============
function initScrollAnimations() {
  // Add animate class to target elements
  const animateTargets = [
    '.stat-card', '.entry-card', '.pipeline-stage',
    '.table-card', '.figure-card', '.rel-card',
    '.demo-card', '.abstract-box', '.category-item',
    '.bench-finding', '.download-section'
  ];

  animateTargets.forEach(selector => {
    document.querySelectorAll(selector).forEach((el, i) => {
      el.classList.add('animate-on-scroll');
      el.style.transitionDelay = `${(i % 4) * 0.1}s`;
    });
  });

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

  document.querySelectorAll('.animate-on-scroll').forEach(el => {
    observer.observe(el);
  });
}

// ============ SMOOTH ACTIVE NAV ============
function initActiveNav() {
  const sections = document.querySelectorAll('section[id], header[id]');
  const navLinks = document.querySelectorAll('.nav-links a');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        navLinks.forEach(link => {
          link.classList.remove('active');
          if (link.getAttribute('href') === `#${id}`) {
            link.classList.add('active');
          }
        });
      }
    });
  }, { threshold: 0.3 });

  sections.forEach(section => observer.observe(section));
}

// ============ NUMBER COUNTER ANIMATION ============
function animateCounter(el, target, duration = 1500) {
  const start = 0;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.floor(start + (target - start) * eased);

    el.textContent = current.toLocaleString();
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      el.textContent = target.toLocaleString();
    }
  }
  requestAnimationFrame(update);
}

// ============ LOAD DEMO DATA FROM FILES ============
async function loadDemoData() {
  const previewMap = {
    users: 'preview-users',
    items: 'preview-items',
    recall: 'preview-recall',
    ranking: 'preview-ranking',
    relevance: 'preview-relevance'
  };

  const files = {
    users: 'demo/users.jsonl',
    items: 'demo/items.jsonl',
    recall: 'demo/recall.jsonl',
    ranking: 'demo/rank.jsonl',
    relevance: 'demo/relevance.jsonl'
  };

  for (const [key, filePath] of Object.entries(files)) {
    try {
      const response = await fetch(filePath);
      if (!response.ok) continue;
      const text = await response.text();
      const lines = text.trim().split('\n').slice(0, 3);
      const samples = lines.map(line => {
        try { return JSON.parse(line); } catch { return null; }
      }).filter(Boolean);

      if (samples.length > 0) {
        const previewEl = document.getElementById(previewMap[key]);
        if (previewEl) {
          const codeEl = previewEl.querySelector('code');
          if (codeEl) {
            codeEl.textContent = JSON.stringify(samples[0], null, 2);
          }
        }
      }
    } catch (err) {
      // Keep static demo content if files can't be loaded
    }
  }
}

// ============ INIT ALL ============
document.addEventListener('DOMContentLoaded', () => {
  createParticles();
  initNavbar();
  initBenchmarkTabs();
  initPreviewTabs();
  initBibtexCopy();
  initScrollAnimations();
  initActiveNav();
  loadDemoData();
});

// Active nav link style
const style = document.createElement('style');
style.textContent = `.nav-links a.active { color: #fff; background: rgba(255,255,255,0.1); }`;
document.head.appendChild(style);
