/**
 * Arty.tsx — Arty, Art Lab's main agent, drawn as an SVG character: a little retro computer — an
 * "interface friend" — whose screen shows his face. He wears a red beret and carries a paintbrush,
 * because he's the studio's artist.
 *
 * Style: vivid flat colours (orange screen, red bezel, sky-blue details), firm black outlines, and a
 * soft grey offset shadow behind him (the `.friend-shadow` CSS class), like a sticker lifted off the
 * page. Everything is plain SVG shapes, so he scales crisply from a 32px avatar to a 200px hero.
 *
 * Arty's `mood` shows what the backend is doing, so the character is part of the "visible" goal:
 *
 *   idle       smiling, holding his paintbrush                   (waiting for you)
 *   thinking   eyes up, "…" on screen, gentle bob                (a model is working)
 *   searching  eyes to the side, holding a magnifying glass      (rag_agent is searching the knowledge base)
 *   happy      a wink and a big open smile                       (an answer just arrived)
 *   blocked    red screen, frowning, holding a stop shield       (the input guard refused the message)
 *
 * Drawing order matters in SVG (later shapes cover earlier ones): feet and the case's side first, then
 * the case, the screen and the face, then the beret and whatever Arty is holding.
 */

export type ArtyMood = 'idle' | 'thinking' | 'searching' | 'happy' | 'blocked'

const INK = '#1b1b1b'
const CASE = '#dfeae8' // pale mint plastic
const CASE_SIDE = '#b7cbc8' // the case's shaded side
const BEZEL = '#ec4b2c' // red frame around the screen
const SCREEN = '#ffb31a' // orange screen
const SCREEN_BLOCKED = '#ff7a5c' // screen turns red-orange when blocked
const SKY = '#1ea7e1'
const BERET = '#ec4b2c'

type Props = {
  mood?: ArtyMood
  size?: number // width and height in pixels
  shadow?: boolean // the grey offset shadow; off for tiny avatars where it would be noise
  className?: string
}

export function Arty({ mood = 'idle', size = 120, shadow = true, className = '' }: Props) {
  // Moods that move get a CSS animation (defined in index.css; switched off for reduced motion).
  const motion = mood === 'thinking' ? 'arty-bob' : mood === 'searching' ? 'arty-sway' : ''
  return (
    <svg
      viewBox="0 0 200 200"
      width={size}
      height={size}
      className={`${motion} ${shadow ? 'friend-shadow' : ''} ${className}`}
      role="img"
      aria-label={`Arty, ${mood}`}
    >
      <g stroke={INK} strokeWidth={5} strokeLinejoin="round" strokeLinecap="round">
        {/* Feet, and the right side of the case (gives him some depth). */}
        <rect x={60} y={160} width={20} height={12} rx={3} fill={SKY} />
        <rect x={120} y={160} width={20} height={12} rx={3} fill={SKY} />
        <path d="M160 64 L174 72 L174 154 L160 162 Z" fill={CASE_SIDE} />

        {/* The case, the red bezel, and the orange screen. */}
        <rect x={36} y={52} width={126} height={112} rx={16} fill={CASE} />
        <rect x={50} y={64} width={98} height={70} rx={10} fill={BEZEL} />
        <rect x={58} y={71} width={82} height={56} rx={8} fill={mood === 'blocked' ? SCREEN_BLOCKED : SCREEN} />
        {/* Screen glare: a short white arc in the top-left corner. */}
        <path d="M66 86 Q66 78 74 78" stroke="#ffffff" strokeWidth={4} fill="none" />

        <Face mood={mood} />

        {/* Control panel: a red knob, three dots, and a blue slot. */}
        <circle cx={58} cy={149} r={6} fill={BEZEL} />
        <g stroke="none" fill={INK}>
          <circle cx={74} cy={149} r={2.6} />
          <circle cx={83} cy={149} r={2.6} />
          <circle cx={92} cy={149} r={2.6} />
        </g>
        <rect x={110} y={144} width={36} height={9} rx={3} fill={SKY} />

        {/* Beret, tilted, with its little stalk. */}
        <path d="M58 58 Q92 16 150 44 Q144 62 58 58 Z" fill={BERET} />
        <path d="M104 30 L108 18" fill="none" />

        <Held mood={mood} />
      </g>
    </svg>
  )
}

/** The face on the screen: eyes, brows and mouth for each mood. */
function Face({ mood }: { mood: ArtyMood }) {
  if (mood === 'happy') {
    // A wink (left eye closed) and a big open smile, like a friendly "got it!".
    return (
      <g>
        <path d="M78 96 Q85 88 92 96" fill="none" />
        <circle cx={112} cy={94} r={6} fill={INK} stroke="none" />
        <path d="M84 106 Q100 124 116 106 Z" fill="#ffffff" />
      </g>
    )
  }
  // Where the pupils sit: up when thinking, to the side when searching, straight ahead otherwise.
  const [dx, dy] = mood === 'thinking' ? [0, -4] : mood === 'searching' ? [4, 0] : [0, 0]
  return (
    <g>
      <circle cx={85 + dx} cy={95 + dy} r={6} fill={INK} stroke="none" />
      <circle cx={113 + dx} cy={95 + dy} r={6} fill={INK} stroke="none" />
      <circle cx={87 + dx} cy={93 + dy} r={1.8} fill="#ffffff" stroke="none" />
      <circle cx={115 + dx} cy={93 + dy} r={1.8} fill="#ffffff" stroke="none" />
      {mood === 'blocked' && (
        <g fill="none" strokeWidth={4}>
          <path d="M76 82 L92 87" />
          <path d="M122 82 L106 87" />
        </g>
      )}
      {mood === 'thinking' && (
        // "…" in the screen's top-right corner.
        <g stroke="none" fill={INK}>
          <circle cx={120} cy={79} r={2.2} />
          <circle cx={127} cy={79} r={2.2} />
          <circle cx={134} cy={79} r={2.2} />
        </g>
      )}
      {mood === 'thinking' ? (
        <circle cx={99} cy={112} r={4} fill="none" strokeWidth={4} />
      ) : mood === 'blocked' ? (
        <path d="M90 114 L108 114" fill="none" />
      ) : (
        <path d="M89 108 Q99 118 109 108" fill="none" />
      )}
    </g>
  )
}

/** What Arty holds on his right: a paintbrush normally, a magnifier while searching, a shield when blocked. */
function Held({ mood }: { mood: ArtyMood }) {
  if (mood === 'searching') {
    return (
      <g>
        <path d="M168 132 L178 116" strokeWidth={9} fill="none" />
        <circle cx={184} cy={102} r={15} fill="#cfe9f8" />
        <path d="M177 97 Q180 91 186 91" stroke="#ffffff" strokeWidth={3} fill="none" />
      </g>
    )
  }
  if (mood === 'blocked') {
    return (
      <g>
        <path d="M166 96 L196 96 L196 118 Q196 134 181 142 Q166 134 166 118 Z" fill="#ffd166" />
        <path d="M174 118 L188 118" fill="none" />
      </g>
    )
  }
  // Paintbrush: a dark outline stroke with a wooden stroke on top, then a sky-blue, paint-loaded tip.
  return (
    <g>
      <path d="M166 138 L188 100" strokeWidth={10} fill="none" />
      <path d="M166 138 L188 100" stroke="#c98b4f" strokeWidth={4} fill="none" />
      <path d="M184 94 Q191 82 198 88 Q196 98 188 103 Z" fill={SKY} />
    </g>
  )
}
