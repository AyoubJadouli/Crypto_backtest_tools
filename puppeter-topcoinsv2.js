// Import puppeteer and other modules
const puppeteer = require('puppeteer');
const fs = require('fs');
const moment = require('moment');

// Define some constants
const DATE_FORMAT = 'YYYY-MM-DD';
const START_DATE = '2023-01-01'; // Change this to your desired start date
const END_DATE = '2023-01-31'; // Change this to your desired end date
const OUTPUT_FILE = 'results.json'; // Change this to your desired output file name

// Define some URLs for scraping
const TWITTER_URL = 'https://twitter.com/search?q=';
const YOUTUBE_URL = 'https://www.youtube.com/results?search_query=';
const MEDIUM_URL = 'https://medium.com/search?q=';
const BITCOINTALK_URL = 'https://bitcointalk.org/index.php?action=search2;search=';
const GOOGLE_NEWS_URL = 'https://news.google.com/search?q=';
const BING_NEWS_URL = 'https://www.bing.com/news/search?q=';

// Define some keywords for searching crypto coins
// Use an array of arrays to specify multiple keywords per coin
// For example: [['bitcoin', 'btc'], ['ethereum', 'eth'], ...]
const KEYWORDS = [['bitcoin', 'btc'], ['ethereum', 'eth'], ['cardano', 'ada'], ['solana', 'sol'], ['dogecoin', 'doge']]; // Add more keywords as you like

// Define a function to scrape a single URL and return the number of results
async function scrapeURL(url) {
  // Launch a browser and create a new page
  const browser =await puppeteer.launch({args: ['--no-sandbox']});

  const page = await browser.newPage();

  // Go to the URL and wait for the page to load
  await page.goto(url, {waitUntil: 'networkidle2'});

  // Find the element that contains the number of results and extract its text content
  let resultCount;
  try {
    const resultElement = await page.$('.result-count'); // This selector may vary depending on the website
    resultCount = await resultElement.evaluate(el => el.textContent);
  } catch (error) {
    console.error(error);
    resultCount = 'N/A'; // If no element is found or an error occurs, return N/A as the result count
  }

  // Close the browser and return the result count
  await browser.close();
  return resultCount;
}

// Define a function to scrape multiple URLs for each keyword and date range and save the results to a file
async function scrapeURLs(urls, keywords, startDate, endDate) {
  // Create an empty object to store the results
  const results = {};

  // Loop through each keyword array (each coin)
  for (let keywordArray of keywords) {
    // Create an empty object to store the results for each coin 
    let coinResults = {};

    // Loop through each keyword in the array (each alias)
    for (let keyword of keywordArray) {
      // Loop through each date from start date to end date (inclusive)
      let currentDate = moment(startDate);
      while (currentDate.isSameOrBefore(endDate)) {
        // Format the current date as YYYY-MM-DD
        let formattedDate =        currentDate.format(DATE_FORMAT);

        // Create an empty object to store the results for each date
        coinResults[formattedDate] = coinResults[formattedDate] || {};

        // Loop through each URL 
        for (let url of urls) {
          // Append the keyword and date parameters to the URL 
          let fullURL;
          if (url === TWITTER_URL) { 
            fullURL =
              url +
              encodeURIComponent(keyword) +
              '&f=tweets&src=typed_query&lf=on' +
              '&since=' +
              formattedDate +
              '&until=' +
              moment(formattedDate).add(1, 'days').format(DATE_FORMAT); 
          } else if (url === YOUTUBE_URL) { 
            fullURL =
              url +
              encodeURIComponent(keyword) +
              '&sp=EgIIAw%253D%253D' + 
              '%2C+after%3A' + 
              formattedDate + 
              '%2C+before%3A' + 
              moment(formattedDate).add(1, 'days').format(DATE_FORMAT);  
          } else if (url === MEDIUM_URL) { 
            fullURL =
              url +
              encodeURIComponent(keyword) +
              '&when=' + formattedDate;  
          } else if (url === BITCOINTALK_URL) { 
            fullURL =
              url +
              encodeURIComponent(keyword) + ';minage=' + formattedDate;  
          } else if (url === GOOGLE_NEWS_URL) { 
            fullURL =
              url +
              encodeURIComponent(keyword) +
              '&hl=en-US&gl=US&ceid=US%3Aen' +
              '&tbs=cdr%3A1%2Ccd_min%3A' + formattedDate + '%2Ccd_max%3A' + formattedDate;  
          } else if (url === BING_NEWS_URL) { 
            fullURL =
              url +
              encodeURIComponent(keyword) +
              '&qft=interval%3D%22' + formattedDate + '..' + formattedDate + '%22';  
          } else {
            fullURL = url; // If none of the above URLs match, use the URL as it is
          }

          // Scrape the full URL and get the result count
          let resultCount = await scrapeURL(fullURL);

          // Store the result count for each URL in the coin results object
          coinResults[formattedDate][url] = coinResults[formattedDate][url] || 0;
          coinResults[formattedDate][url] += Number(resultCount); // Add up the result counts for each alias
        }

        // Increment the current date by one day
        currentDate.add(1, 'days');
      }
    }

    // Store the coin results object in the results object using the first keyword as the key
    results[keywordArray[0]] = coinResults;
  }

  // Write the results object to a JSON file
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));

  // Log a message to indicate that the scraping is done
  console.log('Scraping done. Check ' + OUTPUT_FILE + ' for results.');
}

// Call the scrapeURLs function with the defined constants
scrapeURLs(
  [TWITTER_URL, YOUTUBE_URL, MEDIUM_URL, BITCOINTALK_URL, GOOGLE_NEWS_URL, BING_NEWS_URL],
  KEYWORDS,
  START_DATE,
  END_DATE
);

