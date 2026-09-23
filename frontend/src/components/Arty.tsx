/**
 * Arty.tsx — Arty, Art Lab's main agent, drawn as an SVG character: a pink axolotl artist in a beret.
 *
 * Drawn by hand in the flat "sticker" cartoon style of the rest of the UI: thick black outlines, flat
 * pastel fills, no gradients. Everything is plain SVG shapes, so it scales crisply from a 32px avatar
 * to a 200px hero, and needs no image files.
 *
 * Arty's `mood` shows what the backend is doing, so the character is part of the "visible" goal:
 *
 *   idle       smiling, holding a paintbrush                (waiting for you)
 *   thinking   looking up, "…" bubble, gentle bob           (a model is working)
 *   searching  holding a magnifying glass                   (rag_agent is searching the knowledge base)
 *   happy      eyes closed in a smile                       (an answer just arrived)
 *   blocked    frowning, holding a stop shield              (the input guard refused the message)
 *
 * Drawing order matters in SVG (later shapes cover earlier ones): gills and feet first, then the body,
 * then the face, then the beret and whatever Arty is holding.
 */

export type ArtyMood = 'idle' | 'thinking' | 'searching' | 'happy' | 'blocked'

const INK = '#121212'
const SKIN = '#FFB3C8' // axolotl pink
const GILL = '#F26D95' // darker pink for the gills
const BELLY = '#FFE3EB'
const CHEEK = '#FF7FA3'
const BERET = '#E8485C'

type Props = {
  mood?: ArtyMood
  size?: number // width and height in pixels
  className?: string
}

export function Arty({ mood = 'idle', size = 120, className = '' }: Props) {
  // Moods that move get a CSS animation (defined in index.css; switched off for reduced motion).
  const motion = mood === 'thinking' ? 'arty-bob' : mood === 'searching' ? 'arty-sway' : ''
  return (
    <svg
      viewBox="0 0 200 200"
      width={size}
      height={size}
      className={`${motion} ${className}`}
      role="img"
      aria-label={`Arty the axolotl, ${mood}`}
    >
      <g stroke={INK} strokeWidth={5} strokeLinejoin="round" strokeLinecap="round">
        {/* Gills: three fronds each side, behind the head. */}
        {[
          [38, 78, -35],
          [30, 98, 0],
          [38, 118, 35],
        ].map(([x, y, angle]) => (
          <g key={y}>
            <ellipse cx={x} cy={y} rx={17} ry={7.5} fill={GILL} transform={`rotate(${angle} ${x} ${y})`} />
            <ellipse cx={200 - x} cy={y} rx={17} ry={7.5} fill={GILL} transform={`rotate(${-angle} ${200 - x} ${y})`} />
          </g>
        ))}

        {/* Feet, then the round body-head, then the lighter belly (no outline on the belly). */}
        <ellipse cx={80} cy={170} rx={15} ry={8} fill={SKIN} />
        <ellipse cx={120} cy={170} rx={15} ry={8} fill={SKIN} />
        <ellipse cx={100} cy={112} rx={64} ry={60} fill={SKIN} />
        <ellipse cx={100} cy={142} rx={34} ry={22} fill={BELLY} stroke="none" />

        {/* Cheeks. */}
        <ellipse cx={64} cy={122} rx={10} ry={6} fill={CHEEK} stroke="none" opacity={0.7} />
        <ellipse cx={136} cy={122} rx={10} ry={6} fill={CHEEK} stroke="none" opacity={0.7} />

        <Face mood={mood} />

        {/* Beret, tilted to one side, with its little stalk. */}
        <path d="M64 66 Q92 26 150 52 Q142 70 64 66 Z" fill={BERET} />
        <path d="M108 38 L112 26" fill="none" />

        {/* Paws. */}
        <ellipse cx={58} cy={150} rx={12} ry={9} fill={SKIN} />
        <Held mood={mood} />
        <ellipse cx={144} cy={150} rx={12} ry={9} fill={SKIN} />

        {mood === 'thinking' && (
          <g fill="#FFFFFF">
            <circle cx={160} cy={34} r={16} />
            <circle cx={136} cy={56} r={5} />
            <g stroke="none" fill={INK}>
              <circle cx={152} cy={34} r={2.5} />
              <circle cx={160} cy={34} r={2.5} />
              <circle cx={168} cy={34} r={2.5} />
            </g>
          </g>
        )}
      </g>
    </svg>
  )
}

/** Eyes, brows and mouth for each mood. */
function Face({ mood }: { mood: ArtyMood }) {
  if (mood === 'happy') {
    // Closed, smiling eyes (upside-down arcs) and a wide smile.
    return (
      <g fill="none">
        <path d="M70 106 Q78 96 86 106" />
        <path d="M114 106 Q122 96 130 106" />
        <path d="M86 122 Q100 138 114 122" />
      </g>
    )
  }
  // Pupils look up when thinking; everything else looks straight ahead.
  const lookUp = mood === 'thinking' ? -4 : 0
  return (
    <g>
      <circle cx={78} cy={104 + lookUp} r={7.5} fill={INK} stroke="none" />
      <circle cx={122} cy={104 + lookUp} r={7.5} fill={INK} stroke="none" />
      <circle cx={80.5} cy={101 + lookUp} r={2.4} fill="#FFFFFF" stroke="none" />
      <circle cx={124.5} cy={101 + lookUp} r={2.4} fill="#FFFFFF" stroke="none" />
      {mood === 'blocked' && (
        <g fill="none">
          <path d="M68 88 L86 94" />
          <path d="M132 88 L114 94" />
        </g>
      )}
      {mood === 'thinking' ? (
        <circle cx={100} cy={126} r={4.5} fill="none" />
      ) : mood === 'blocked' ? (
        <path d="M89 128 L111 128" fill="none" />
      ) : (
        <path d="M89 122 Q100 132 111 122" fill="none" />
      )}
    </g>
  )
}

/** What Arty holds in his right paw: a brush normally, a magnifier while searching, a shield when blocked. */
function Held({ mood }: { mood: ArtyMood }) {
  if (mood === 'searching') {
    return (
      <g>
        <path d="M150 146 L164 126" strokeWidth={9} fill="none" />
        <circle cx={172} cy={112} r={17} fill="#CFE6FF" />
        <path d="M164 106 Q168 100 175 100" stroke="#FFFFFF" strokeWidth={4} fill="none" />
      </g>
    )
  }
  if (mood === 'blocked') {
    return (
      <g>
        <path d="M150 104 L184 104 L184 128 Q184 146 167 154 Q150 146 150 128 Z" fill="#FFD166" />
        <path d="M160 128 L174 128" fill="none" />
      </g>
    )
  }
  // Paintbrush: a dark outline stroke with a wooden stroke on top, then a blue, paint-loaded tip.
  return (
    <g>
      <path d="M148 150 L176 106" strokeWidth={10} fill="none" />
      <path d="M148 150 L176 106" stroke="#C98B4F" strokeWidth={4} fill="none" />
      <path d="M172 100 Q180 88 188 94 Q186 104 176 110 Z" fill="#5B8DEF" />
    </g>
  )
}
