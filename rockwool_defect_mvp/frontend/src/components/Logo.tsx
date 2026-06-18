export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 60 60" className={className} xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <defs>
        <linearGradient id="lg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#E11D2A" />
          <stop offset="1" stopColor="#B81522" />
        </linearGradient>
      </defs>
      <rect x="6" y="6" width="48" height="48" rx="10" transform="rotate(45 30 30)" fill="url(#lg)" />
      <rect x="13" y="13" width="34" height="34" rx="7" transform="rotate(45 30 30)" fill="#1E2A6B" />
      <text x="30" y="38" textAnchor="middle" fontFamily="Geist, sans-serif" fontWeight="900" fontSize="22" fill="#fff">M</text>
    </svg>
  );
}
