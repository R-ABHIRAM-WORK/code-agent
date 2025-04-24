const excuses = [
    "My alarm clock is currently in a philosophical debate with itself.",
    "I was abducted by squirrels. They demanded nuts.",
    "A rogue Roomba blocked my exit.",
    "My pet goldfish was feeling lonely, so I had to read it poetry.",
    "I accidentally joined a parade and couldn't get out.",
    "My shadow ran away and I had to chase it.",
    "I was trying to teach my cat to yodel.",
    "A flock of pigeons stole my car keys.",
    "My Wi-Fi password changed itself and held my schedule hostage.",
    "I got stuck in a time loop, this is attempt #37.",
    "My socks were mismatched, causing a critical balance issue.",
    "I was held up by a very slow-moving sloth crossing the road.",
    "My coffee wasn't strong enough to face the day yet.",
    "I mistook my toothpaste for hair gel.",
    "A wizard turned my doorknob into a banana.",
    "I was busy explaining the internet to my grandparents.",
    "My reflection wouldn't follow me this morning.",
    "I fell into a YouTube rabbit hole about competitive dog grooming.",
    "The existential dread was particularly heavy today.",
    "I had to wrestle a goose for my lunch money."
];

const excuseDisplay = document.getElementById('excuse-display');
const generateBtn = document.getElementById('generate-btn');
const copyBtn = document.getElementById('copy-btn');
const shareBtn = document.getElementById('share-btn');
const copyFeedback = document.getElementById('copy-feedback');

function getRandomExcuse() {
    const randomIndex = Math.floor(Math.random() * excuses.length);
    return excuses[randomIndex];
}

function displayNewExcuse() {
    excuseDisplay.textContent = getRandomExcuse();
    copyFeedback.textContent = ''; // Clear feedback on new excuse
}

function copyToClipboard() {
    const excuseText = excuseDisplay.textContent;
    navigator.clipboard.writeText(excuseText).then(() => {
        copyFeedback.textContent = 'Copied!';
        // Optional: Clear feedback after a few seconds
        setTimeout(() => { copyFeedback.textContent = ''; }, 2000);
    }).catch(err => {
        console.error('Failed to copy: ', err);
        copyFeedback.textContent = 'Copy failed!';
    });
}

function shareOnTwitter() {
    const excuseText = excuseDisplay.textContent;
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent('My excuse for being late: "' + excuseText + '" #LateExcuse #Funny')}`;
    window.open(twitterUrl, '_blank');
}

// Event Listeners
generateBtn.addEventListener('click', displayNewExcuse);
copyBtn.addEventListener('click', copyToClipboard);
shareBtn.addEventListener('click', shareOnTwitter);

// Initial excuse on page load
displayNewExcuse();
