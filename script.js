// NFL Kalshi Trader JavaScript

// Configuration
const KALSHI_API_KEY = '8b0f33c5-1607-4b0e-861f-6b6fa9e64ecd';
const KALSHI_API_BASE = '/api/kalshi'; // Use local proxy server

// Global state
let currentMarkets = [];
let userBalance = 0;
let selectedTrade = null;

// DOM Elements
const loadingSpinner = document.getElementById('loading-spinner');
const gamesSection = document.getElementById('games-section');
const gamesContainer = document.getElementById('games-container');
const refreshBtn = document.getElementById('refresh-btn');
const accountBalance = document.getElementById('account-balance');
const tradeModal = document.getElementById('trade-modal');
const closeModal = document.getElementById('close-modal');
const cancelTradeBtn = document.getElementById('cancel-trade-btn');
const placeTradeBtn = document.getElementById('place-trade-btn');

// API Functions
async function fetchWithAuth(url, options = {}) {
    try {
        console.log(`🌐 Making API request to: ${url}`);
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        console.log(`📡 Response status: ${response.status} ${response.statusText}`);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error(`❌ API Error Response: ${errorText}`);
            throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        console.log('✅ API request successful');
        return data;
    } catch (error) {
        console.error('❌ API request failed:', error);
        showNotification(`API Error: ${error.message}`, 'error');
        return null;
    }
}

async function fetchAccountBalance() {
    try {
        const data = await fetchWithAuth(`${KALSHI_API_BASE}/portfolio/balance`);
        if (data && data.balance !== undefined) {
            userBalance = data.balance / 100; // Convert cents to dollars
            accountBalance.textContent = `$${userBalance.toFixed(2)}`;
        } else {
            // Fallback for demo purposes
            userBalance = 1000;
            accountBalance.textContent = `$${userBalance.toFixed(2)}`;
        }
    } catch (error) {
        console.error('Failed to fetch balance:', error);
        // Demo balance
        userBalance = 1000;
        accountBalance.textContent = `$${userBalance.toFixed(2)}`;
    }
}

async function fetchNFLMarkets() {
    try {
        showLoading(true);
        console.log('🔄 Attempting to fetch markets from Kalshi API...');
        
        // Fetch markets with NFL filter
        const marketsData = await fetchWithAuth(`${KALSHI_API_BASE}/markets?limit=100&status=open`);
        
        console.log('📊 Raw API response:', marketsData);
        
        if (!marketsData || !marketsData.markets) {
            throw new Error('No markets data received');
        }
        
        console.log(`📈 Total markets received: ${marketsData.markets.length}`);
        
        // Log ALL markets first
        console.log('🔍 ALL MARKETS FROM KALSHI:');
        marketsData.markets.forEach((market, index) => {
            console.log(`${index + 1}. ${market.ticker} - ${market.title}`);
        });
        
        // Filter for NFL moneyline markets (single games, not collections)
        const nflMarkets = marketsData.markets.filter(market => {
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
        
        console.log(`🏈 NFL markets found: ${nflMarkets.length}`);
        if (nflMarkets.length > 0) {
            console.log('🏈 NFL MARKETS DETAILS:');
            nflMarkets.forEach((market, index) => {
                console.log(`${index + 1}. ${market.ticker} - ${market.title}`);
                console.log(`   Status: ${market.status}, Volume: ${market.volume || 'N/A'}`);
                console.log(`   Yes Ask: ${market.yes_ask}¢, No Ask: ${market.no_ask}¢`);
            });
        } else {
            console.log('⚠️ No NFL-related markets found in the response');
        }
        
        if (nflMarkets.length === 0) {
            console.log('⚠️ No NFL markets found');
            showNotification('No NFL markets found on Kalshi', 'warning');
            currentMarkets = [];
        } else {
            console.log('✅ Using real Kalshi NFL markets');
            showNotification(`Found ${nflMarkets.length} NFL markets from Kalshi!`, 'success');
            currentMarkets = nflMarkets;
        }
        
        renderMarkets();
        showLoading(false);
        
    } catch (error) {
        console.error('❌ Failed to fetch NFL markets:', error);
        console.log('🔄 No fallback - showing error message');
        currentMarkets = [];
        renderMarkets();
        showLoading(false);
        showNotification('Failed to connect to Kalshi API - No markets available', 'error');
    }
}

// Demo markets function removed - only using real Kalshi API data

function renderMarkets() {
    gamesContainer.innerHTML = '';
    
    if (currentMarkets.length === 0) {
        // Show empty state
        const emptyState = document.createElement('div');
        emptyState.className = 'empty-state';
        emptyState.innerHTML = `
            <div class="empty-state-content">
                <h3>No NFL Markets Available</h3>
                <p>No NFL markets are currently available on Kalshi.</p>
                <p>Try refreshing or check back later.</p>
                <button id="retry-btn" class="refresh-btn">🔄 Try Again</button>
            </div>
        `;
        gamesContainer.appendChild(emptyState);
        
        // Add retry functionality
        const retryBtn = document.getElementById('retry-btn');
        retryBtn.addEventListener('click', () => {
            fetchNFLMarkets();
            fetchAccountBalance();
        });
    } else {
        currentMarkets.forEach(market => {
            const gameCard = createGameCard(market);
            gamesContainer.appendChild(gameCard);
        });
    }
    
    gamesSection.style.display = 'block';
}

function createGameCard(market) {
    const card = document.createElement('div');
    card.className = 'game-card';
    
    // Extract team names from title
    const teams = extractTeamNames(market.title);
    
    card.innerHTML = `
        <div class="game-header">
            <h3 class="game-title">${teams.team1} vs ${teams.team2}</h3>
            <p class="game-subtitle">${market.subtitle || formatDateTime(market.close_time)}</p>
        </div>
        <div class="game-body">
            <div class="market-info">
                <p><strong>Market:</strong> ${market.title}</p>
                <p><strong>Volume:</strong> ${market.volume ? market.volume.toLocaleString() : 'N/A'} contracts</p>
                <p><strong>Closes:</strong> ${formatDateTime(market.close_time)}</p>
            </div>
            <div class="teams-container">
                <div class="team-option" data-ticker="${market.ticker}" data-side="yes">
                    <div class="team-name">${teams.team1}</div>
                    <div class="team-price">${market.yes_ask || 50}¢</div>
                </div>
                <div class="vs-divider">VS</div>
                <div class="team-option" data-ticker="${market.ticker}" data-side="no">
                    <div class="team-name">${teams.team2}</div>
                    <div class="team-price">${market.no_ask || 50}¢</div>
                </div>
            </div>
        </div>
    `;
    
    // Add click handlers for team options
    const teamOptions = card.querySelectorAll('.team-option');
    teamOptions.forEach(option => {
        option.addEventListener('click', () => {
            const ticker = option.dataset.ticker;
            const side = option.dataset.side;
            const price = parseInt(option.querySelector('.team-price').textContent);
            openTradeModal(market, side, price);
        });
    });
    
    return card;
}

function extractTeamNames(title) {
    // Simple extraction - in a real app, you'd have a more robust parser
    const match = title.match(/Will (.+?) beat (.+?)\?/);
    if (match) {
        return {
            team1: match[1].trim(),
            team2: match[2].trim()
        };
    }
    
    // Fallback
    return {
        team1: 'Team A',
        team2: 'Team B'
    };
}

function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        timeZoneName: 'short'
    });
}

function openTradeModal(market, side, price) {
    selectedTrade = { market, side, price };
    
    // Populate modal
    document.getElementById('trade-market').textContent = market.title;
    document.getElementById('trade-contract').textContent = side === 'yes' ? 'Yes' : 'No';
    document.getElementById('trade-price').textContent = price;
    document.getElementById('trade-price-input').value = price;
    
    // Show modal
    tradeModal.classList.add('show');
    
    // Update trade summary
    updateTradeSummary();
    
    // Add event listeners for inputs
    const quantityInput = document.getElementById('trade-quantity');
    const priceInput = document.getElementById('trade-price-input');
    
    quantityInput.addEventListener('input', updateTradeSummary);
    priceInput.addEventListener('input', updateTradeSummary);
}

function updateTradeSummary() {
    const quantity = parseInt(document.getElementById('trade-quantity').value) || 0;
    const price = parseInt(document.getElementById('trade-price-input').value) || 0;
    
    const totalCost = (quantity * price) / 100; // Convert cents to dollars
    const maxProfit = (quantity * (100 - price)) / 100;
    
    document.getElementById('total-cost').textContent = `$${totalCost.toFixed(2)}`;
    document.getElementById('max-profit').textContent = `$${maxProfit.toFixed(2)}`;
    
    // Enable/disable place trade button
    const placeBtn = document.getElementById('place-trade-btn');
    placeBtn.disabled = totalCost > userBalance || quantity <= 0 || price <= 0;
}

async function placeTrade() {
    if (!selectedTrade) return;
    
    const quantity = parseInt(document.getElementById('trade-quantity').value);
    const price = parseInt(document.getElementById('trade-price-input').value);
    const action = document.getElementById('trade-action').value;
    
    try {
        placeTradeBtn.disabled = true;
        placeTradeBtn.textContent = 'Placing Trade...';
        
        // Make real API call to place trade
        const result = await fetchWithAuth(`${KALSHI_API_BASE}/portfolio/orders`, {
            method: 'POST',
            body: JSON.stringify({
                ticker: selectedTrade.market.ticker,
                side: selectedTrade.side,
                action: action,
                count: quantity,
                price: price
            })
        });
        
        if (!result) {
            throw new Error('Failed to place trade - no response from API');
        }
        
        console.log('✅ Trade placed successfully:', result);
        
        showNotification(`Trade placed successfully! ${quantity} contracts at ${price}¢`, 'success');
        closeTradeModal();
        
        // Update balance (simulation)
        const totalCost = (quantity * price) / 100;
        userBalance -= totalCost;
        accountBalance.textContent = `$${userBalance.toFixed(2)}`;
        
    } catch (error) {
        console.error('Failed to place trade:', error);
        showNotification(`Failed to place trade: ${error.message}`, 'error');
    } finally {
        placeTradeBtn.disabled = false;
        placeTradeBtn.textContent = 'Place Trade';
    }
}

function closeTradeModal() {
    tradeModal.classList.remove('show');
    selectedTrade = null;
}

function showLoading(show) {
    loadingSpinner.style.display = show ? 'flex' : 'none';
    gamesSection.style.display = show ? 'none' : 'block';
}

function showNotification(message, type = 'info') {
    // Remove existing notification
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    // Create new notification
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    
    // Color based on type
    let backgroundColor = '#3b82f6';
    if (type === 'success') backgroundColor = '#10b981';
    if (type === 'error') backgroundColor = '#ef4444';
    if (type === 'warning') backgroundColor = '#f59e0b';
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        background: backgroundColor,
        color: 'white',
        padding: '1rem 1.5rem',
        borderRadius: '0.5rem',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        zIndex: '1001',
        transform: 'translateX(100%)',
        transition: 'transform 0.3s ease',
        maxWidth: '400px',
        fontSize: '0.875rem',
        fontWeight: '500'
    });
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Animate out and remove
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

// Event Listeners
refreshBtn.addEventListener('click', () => {
    fetchNFLMarkets();
    fetchAccountBalance();
});

closeModal.addEventListener('click', closeTradeModal);
cancelTradeBtn.addEventListener('click', closeTradeModal);
placeTradeBtn.addEventListener('click', placeTrade);

// Close modal when clicking outside
tradeModal.addEventListener('click', (e) => {
    if (e.target === tradeModal) {
        closeTradeModal();
    }
});

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🏈 NFL Kalshi Trader loaded!');
    console.log('🔑 Using API Key:', KALSHI_API_KEY);
    console.log('🌐 API Base URL: https://api.elections.kalshi.com/trade-api/v2 (via proxy)');
    
    // Load initial data
    console.log('💰 Fetching account balance...');
    await fetchAccountBalance();
    
    console.log('📊 Fetching NFL markets...');
    await fetchNFLMarkets();
    
    showNotification('Welcome to NFL Kalshi Trader! 🏈', 'success');
});
