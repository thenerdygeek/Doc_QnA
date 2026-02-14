/**
 * Mermaid diagram syntax validator using @a24z/mermaid-parser.
 *
 * Reads diagram text from stdin, outputs JSON validation result to stdout.
 *
 * Setup:
 *   cd scripts && npm install
 *
 * Usage:
 *   echo "graph TD; A-->B" | node validate_mermaid.mjs
 *
 * Output:
 *   {"valid": true,  "type": "flowchart"}
 *   {"valid": false, "error": "..."}
 */

import { parse } from '@a24z/mermaid-parser';

let data = '';
process.stdin.setEncoding('utf8');
for await (const chunk of process.stdin) {
  data += chunk;
}

try {
  const result = await parse(data.trim());
  if (result.valid) {
    console.log(JSON.stringify({ valid: true, type: result.type || null }));
  } else {
    console.log(
      JSON.stringify({
        valid: false,
        type: result.type || null,
        error: result.error || 'Unknown syntax error',
      }),
    );
  }
} catch (e) {
  console.log(JSON.stringify({ valid: false, error: e.message }));
}
