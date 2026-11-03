// The voice command parser: phrases a player might say, and chatter that must not be a command.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseCommand } from '../../js/voice.js';

const show = c => c ? `${c.type}${c.value === undefined ? '' : ':' + (typeof c.value === 'object' && c.value !== null ? JSON.stringify(c.value) : c.value)}` : null;
const cases = [
  ['Start recording', 'record:start'], ['record', 'record:start'], ['OK Kinesphere, start recording please', 'record:start'],
  ['stop', 'stop'], ['Stop recording.', 'stop'], ['finish', 'stop'], ['cancel', 'record:cancel'],
  ['neon', 'style:neon'], ['Switch to pencil sketch', 'style:pencil'], ['origami style', 'style:origami'], ['sandman', 'style:sandman'],
  ['sand man', 'style:sandman'], ['fuzzy blur', 'style:fuzzy'], ['stick figure', 'style:stickman'], ['light trails', 'style:trails'],
  ['next style', 'style:next'], ['skeleton', 'style:skeleton'], ['constellation', 'style:constellation'], ['use paper', 'style:origami'],
  ['hoops', 'style:hoops'], ['hula hoop', 'style:hoops'], ['animator', 'style:animator'], ['pleats', 'style:pleats'],
  ['overlay off', 'overlay:false'], ['hide the overlay', 'overlay:false'], ['show overlay', 'overlay:true'],
  ['hide video', 'video:false'], ['camera feed on', 'video:true'], ['turn the video off', 'video:false'],
  ['full screen', 'fullscreen:true'], ['exit fullscreen', 'fullscreen:false'],
  ['mirror off', 'mirror:false'], ['flip', 'mirror:null'], ['3D view', 'view3d:true'], ['close three d', 'view3d:false'], ['show 3d', 'view3d:true'],
  ['smoothing off', 'smoothing:off'], ['strong smoothing', 'smoothing:strong'], ['smoothing normal', 'smoothing:normal'],
  ['use multipose', 'model:movenet-multipose'], ['model heavy', 'model:heavy'], ['switch to lightning', 'model:movenet-lightning'], ['mediapipe lite', 'model:lite'], ['holistic model', 'model:holistic'],
  ['use lite', 'model:lite'], ['switch to heavy', 'model:heavy'], ['use light trails', 'style:trails'],
  ['two people', 'people:2'], ['track three dancers', 'people:3'], ['just me', 'people:1'], ['one person', 'people:1'], ['to people', 'people:2'],
  ['blue sand', 'sand:{"hue":215}'], ['colour pink', 'sand:{"hue":325}'], ['make it green', 'sand:{"hue":125}'], ['red particles', 'sand:{"hue":0}'], ['blue', 'sand:{"hue":215}'],
  ['gravity up', 'sand:{"gravity":0.5,"relative":true}'], ['gravity off', 'sand:{"gravity":0}'], ['gravity two', 'sand:{"gravity":2}'], ['float', 'sand:{"gravity":-0.6}'], ['anti gravity', 'sand:{"gravity":-0.6}'],
  ['longer trails', 'sand:{"decay":1,"relative":true}'], ['decay five seconds', 'sand:{"decay":5}'], ['shorter decay', 'sand:{"decay":-1,"relative":true}'],
  ['more momentum', 'sand:{"momentum":0.5,"relative":true}'], ['momentum off', 'sand:{"momentum":0}'], ['reset sand', 'sand:reset'],
  ['go to the dashboard', 'nav:dashboard'], ['sessions', 'nav:sessions'], ['game', 'nav:game'], ['open game mode', 'nav:game'], ['live view', 'nav:live'],
  ['play sunrise', 'game:play:sunrise'], ['play neon nights', 'game:play:neon'], ['play the hard song', 'game:play:rush'], ['play a game', 'nav:game'],
  ['stop the song', 'game:stop'], ['louder', 'game:volume:0.15'], ['quieter', 'game:volume:-0.15'], ['mute', 'game:volume:0'],
  ['help', 'help:show'], ['what can I say', 'help:show'], ['close', 'help:hide'], ['stop listening', 'voice:off'],
  ['invite', 'mp:invite'], ['leave the session', 'mp:leave'],
];
for (const [phrase, want] of cases) test(`"${phrase}" → ${want}`, () => assert.equal(show(parseCommand(phrase)), want));
for (const phrase of ['I think that went really well', 'what time is it', 'my arms are tired', 'let me try that again', ''])
  test(`"${phrase}" is not a command`, () => assert.equal(parseCommand(phrase), null));
