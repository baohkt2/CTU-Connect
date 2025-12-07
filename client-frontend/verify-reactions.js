#!/usr/bin/env node

/**
 * Script để verify reactions đã được update đúng chưa
 * Updated to check PostCard.tsx (root cause)
 */

const fs = require('fs');
const path = require('path');

const EXPECTED_REACTIONS = ['LIKE', 'INSIGHTFUL', 'RELEVANT', 'USEFUL_SOURCE', 'QUESTION'];
const OLD_REACTIONS = ['LOVE', 'HAHA', 'WOW', 'SAD', 'ANGRY'];
const OLD_EMOJIS = ['❤️', '😂', '😮', '😢', '😡'];

console.log('\n🔍 Verifying Reaction Updates...\n');

let hasErrors = false;

// Check ReactionPicker.tsx
const pickerPath = path.join(__dirname, 'src/components/ui/ReactionPicker.tsx');
const pickerContent = fs.readFileSync(pickerPath, 'utf8');

console.log('📄 Checking ReactionPicker.tsx:');

EXPECTED_REACTIONS.forEach(reaction => {
  if (pickerContent.includes(`id: '${reaction}'`)) {
    console.log(`  ✅ Found: ${reaction}`);
  } else {
    console.log(`  ❌ Missing: ${reaction}`);
    hasErrors = true;
  }
});

OLD_REACTIONS.forEach(reaction => {
  if (pickerContent.includes(`id: '${reaction}'`)) {
    console.log(`  ⚠️  Old reaction still exists: ${reaction}`);
    hasErrors = true;
  }
});

if (!hasErrors) {
  console.log('  ✅ All correct!\n');
}

// Check types/index.ts
const typesPath = path.join(__dirname, 'src/types/index.ts');
const typesContent = fs.readFileSync(typesPath, 'utf8');

console.log('📄 Checking types/index.ts:');

const enumMatch = typesContent.match(/export enum ReactionType \{([^}]+)\}/s);
if (enumMatch) {
  const enumContent = enumMatch[1];
  
  EXPECTED_REACTIONS.forEach(reaction => {
    if (enumContent.includes(reaction)) {
      console.log(`  ✅ Enum has: ${reaction}`);
    } else {
      console.log(`  ❌ Enum missing: ${reaction}`);
      hasErrors = true;
    }
  });
  
  OLD_REACTIONS.forEach(reaction => {
    if (enumContent.includes(`${reaction} =`) || enumContent.includes(`${reaction},`)) {
      console.log(`  ⚠️  Enum still has old reaction: ${reaction}`);
      hasErrors = true;
    }
  });
}

if (!hasErrors) {
  console.log('  ✅ All correct!\n');
}

// Check PostCard.tsx (ROOT CAUSE!)
const postCardPath = path.join(__dirname, 'src/components/post/PostCard.tsx');
const postCardContent = fs.readFileSync(postCardPath, 'utf8');

console.log('📄 Checking PostCard.tsx (CRITICAL):');

// Check for REACTIONS import
if (postCardContent.includes("import { REACTIONS } from '@/components/ui/ReactionPicker'") ||
    postCardContent.includes('import { REACTIONS } from "@/components/ui/ReactionPicker"')) {
  console.log('  ✅ Imports REACTIONS from ReactionPicker');
} else {
  console.log('  ❌ Missing import: REACTIONS from ReactionPicker');
  hasErrors = true;
}

// Check for hardcoded old emojis
const hasHardcodedEmojis = OLD_EMOJIS.some(emoji => postCardContent.includes(`'${emoji}'`));
if (hasHardcodedEmojis) {
  console.log('  ⚠️  Still has hardcoded old emojis!');
  hasErrors = true;
} else {
  console.log('  ✅ No hardcoded old emojis');
}

// Check for REACTIONS.map usage
if (postCardContent.includes('REACTIONS.map')) {
  console.log('  ✅ Uses REACTIONS.map() (correct)');
} else {
  console.log('  ❌ Not using REACTIONS.map()');
  hasErrors = true;
}

// Check for reaction.id usage
if (postCardContent.includes('reaction.id') && postCardContent.includes('handleReactionClick(reaction.id)')) {
  console.log('  ✅ Passes reaction.id to handler (correct)');
} else {
  console.log('  ⚠️  May not be passing correct reaction IDs');
  hasErrors = true;
}

// Summary
console.log('\n' + '='.repeat(60));
if (hasErrors) {
  console.log('❌ VERIFICATION FAILED - Please check the issues above');
  console.log('\n🔧 To fix:');
  console.log('1. Make sure PostCard.tsx imports REACTIONS');
  console.log('2. Remove hardcoded emoji arrays');
  console.log('3. Use REACTIONS.map() instead of hardcode');
  console.log('4. Clear cache: rm -rf .next .swc');
  console.log('5. Restart dev server: npm run dev');
  console.log('6. Hard refresh browser: Ctrl+Shift+R');
  process.exit(1);
} else {
  console.log('✅ ALL VERIFICATIONS PASSED!');
  console.log('\n🚀 Next steps:');
  console.log('1. Clear cache: rm -rf .next .swc');
  console.log('2. Start server: npm run dev');
  console.log('3. Hard refresh browser: Ctrl+Shift+R');
  console.log('4. Test hover on Like button');
  console.log('\n🎉 You should see 5 NEW reactions now!');
  process.exit(0);
}
