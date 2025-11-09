const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;

// Kalshi API configuration
const KALSHI_API_KEY = '8b0f33c5-1607-4b0e-861f-6b6fa9e64ecd';
const KALSHI_API_BASE = 'https://api.elections.kalshi.com/trade-api/v2';

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('.'));

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Proxy endpoint for Kalshi API requests
app.get('/api/kalshi/*', async (req, res) => {
    try {
        const kalshiPath = req.params[0];
        const queryString = req.url.split('?')[1] || '';
        const kalshiUrl = `${KALSHI_API_BASE}/${kalshiPath}${queryString ? '?' + queryString : ''}`;
        
        console.log(`🌐 Proxying request to: ${kalshiUrl}`);
        
        const fetch = (await import('node-fetch')).default;
        const response = await fetch(kalshiUrl, {
            method: req.method,
            headers: {
                'Authorization': `Bearer ${KALSHI_API_KEY}`,
                'Content-Type': 'application/json',
                'User-Agent': 'NFL-Kalshi-Trader/1.0'
            }
        });
        
        console.log(`📡 Kalshi response status: ${response.status} ${response.statusText}`);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error(`❌ Kalshi API error: ${errorText}`);
            return res.status(response.status).json({
                error: `Kalshi API error: ${response.status} ${response.statusText}`,
                details: errorText
            });
        }
        
        const data = await response.json();
        console.log(`✅ Successfully proxied Kalshi request`);
        
        // Log market data if it's the markets endpoint
        if (kalshiPath.includes('markets')) {
            console.log(`📊 Markets received: ${data.markets ? data.markets.length : 0}`);
            if (data.markets) {
                console.log('🔍 ALL MARKETS FROM KALSHI:');
                data.markets.forEach((market, index) => {
                    console.log(`${index + 1}. ${market.ticker} - ${market.title}`);
                    console.log(`   Category: ${market.category || 'N/A'}, Subtitle: ${market.subtitle || 'N/A'}`);
                });
                
                // Filter for NFL moneyline markets (single games, not collections)
                const nflMarkets = data.markets.filter(market => {
                    const title = (market.title || '').toLowerCase();
                    const ticker = (market.ticker || '').toLowerCase();
                    const subtitle = (market.subtitle || '').toLowerCase();
                    
                    // Look for NFL/football markets
                    const isNFL = title.includes('nfl') || 
                                 title.includes('football') || 
                                 ticker.includes('nfl') ||
                                 subtitle.includes('nfl');
                    
                    // Filter for moneyline markets (single game outcomes)
                    const isMoneyline = title.includes('win') || 
                                       title.includes('beat') || 
                                       title.includes('defeat') ||
                                       title.includes('moneyline') ||
                                       title.includes('winner');
                    
                    // Exclude collections/parlays
                    const isNotCollection = !title.includes('collection') && 
                                           !title.includes('parlay') &&
                                           !title.includes('multiple') &&
                                           !ticker.includes('coll');
                    
                    return isNFL && isMoneyline && isNotCollection;
                });
                
                console.log(`🏈 NFL moneyline markets found: ${nflMarkets.length}`);
                if (nflMarkets.length > 0) {
                    console.log('🏈 NFL MONEYLINE MARKETS DETAILS:');
                    nflMarkets.forEach((market, index) => {
                        console.log(`${index + 1}. ${market.ticker} - ${market.title}`);
                        console.log(`   Status: ${market.status}, Volume: ${market.volume || 'N/A'}`);
                        console.log(`   Category: ${market.category || 'N/A'}, Subtitle: ${market.subtitle || 'N/A'}`);
                    });
                } else {
                    console.log('⚠️ No NFL moneyline markets found. Showing some sample markets for debugging:');
                    data.markets.slice(0, 5).forEach((market, index) => {
                        console.log(`Sample ${index + 1}: ${market.ticker} - ${market.title}`);
                    });
                }
            }
        }
        
        res.json(data);
        
    } catch (error) {
        console.error('❌ Proxy error:', error);
        res.status(500).json({
            error: 'Proxy server error',
            details: error.message
        });
    }
});

// Handle POST requests for trading
app.post('/api/kalshi/*', async (req, res) => {
    try {
        const kalshiPath = req.params[0];
        const kalshiUrl = `${KALSHI_API_BASE}/${kalshiPath}`;
        
        console.log(`🌐 Proxying POST request to: ${kalshiUrl}`);
        console.log(`📝 Request body:`, req.body);
        
        const fetch = (await import('node-fetch')).default;
        const response = await fetch(kalshiUrl, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${KALSHI_API_KEY}`,
                'Content-Type': 'application/json',
                'User-Agent': 'NFL-Kalshi-Trader/1.0'
            },
            body: JSON.stringify(req.body)
        });
        
        console.log(`📡 Kalshi response status: ${response.status} ${response.statusText}`);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error(`❌ Kalshi API error: ${errorText}`);
            return res.status(response.status).json({
                error: `Kalshi API error: ${response.status} ${response.statusText}`,
                details: errorText
            });
        }
        
        const data = await response.json();
        console.log(`✅ Successfully proxied Kalshi POST request`);
        
        res.json(data);
        
    } catch (error) {
        console.error('❌ Proxy POST error:', error);
        res.status(500).json({
            error: 'Proxy server error',
            details: error.message
        });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`🏈 NFL Kalshi Trader server running at http://localhost:${PORT}`);
    console.log(`🔑 Using Kalshi API Key: ${KALSHI_API_KEY}`);
    console.log(`🌐 Proxying requests to: ${KALSHI_API_BASE}`);
    console.log(`📊 Server will log all market data when fetched`);
});
