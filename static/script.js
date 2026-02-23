// API Configuration
const API = {
    symbols: '/api/symbols',
    summary: '/api/summary',
    signals: '/api/signals',
    indicators: '/api/indicators',
    symbolIndicators: (symbol) => `/api/symbol/${encodeURIComponent(symbol)}/indicators`,
    symbolChart: (symbol) => `/api/symbol/${encodeURIComponent(symbol)}/chart`
};

// Pagination State
let currentPage = 1;
let pageSize = 50;
let allSignals = [];

// ============================================
// UTILITY FUNCTIONS
// ============================================
function formatSignal(signal) {
    if (signal === 'BUY') return '<span class="signal-badge signal-buy">Buy</span>';
    return '<span class="signal-badge signal-notbuy">Not-Buy</span>';
}

function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString();
}

function showError(message) {
    console.error(message);
    showNotification(message, 'error', 'Error');
}

function showNotification(message, type = 'info', title = '') {
    const container = document.getElementById('notificationContainer');
    if (!container) return;
    
    const icons = {
        info: 'ℹ️',
        success: '✓',
        error: '✕',
        warning: '⚠'
    };
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.position = 'relative';
    notification.innerHTML = `
        <div class="notification-icon">${icons[type] || icons.info}</div>
        <div class="notification-content">
            ${title ? `<div class="notification-title">${title}</div>` : ''}
            <div class="notification-message">${message}</div>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">×</button>
        <div class="notification-progress"></div>
    `;
    
    container.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentElement) {
            notification.classList.add('removing');
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 300);
        }
    }, 1500);
}

// ============================================
// LATEST DAY BUY SIGNALS
// ============================================
async function loadLatestSignals() {
    try {
        console.log('📊 [SIGNALS] Loading latest day BUY signals...');
        
        const response = await fetch(API.signals);
        const data = await response.json();
        
        console.log(`📊 [SIGNALS] Found ${data.length} BUY signals for latest day`);
        
        displayLatestSignals(data);
        
    } catch (error) {
        console.error('❌ [SIGNALS] Failed to load signals:', error);
    }
}

function displayLatestSignals(signals) {
    const container = document.getElementById('signalsContainer');
    if (!container) return;
    
    if (signals.length === 0) {
        container.innerHTML = '<div class="empty-state">No BUY signals available for latest day</div>';
        return;
    }
    
    // Get the date from first signal
    const signalDate = new Date(signals[0].date);
    
    let html = `
        <div class="signals-header">
            <div class="signals-info">
                <h3>Latest Day BUY Signals</h3>
                <p class="signals-date">Date: ${signalDate.toLocaleDateString()}</p>
            </div>
            <div class="signals-count">${signals.length} BUY signals</div>
        </div>
    `;
    
    // Group signals by indicator
    const signalsByIndicator = {};
    signals.forEach(sig => {
        if (!signalsByIndicator[sig.indicator]) {
            signalsByIndicator[sig.indicator] = [];
        }
        signalsByIndicator[sig.indicator].push(sig);
    });
    
    for (const [indicator, indicatorSignals] of Object.entries(signalsByIndicator)) {
        html += `
            <div class="indicator-group">
                <div class="indicator-title">${indicator} (${indicatorSignals.length} BUY signals)</div>
                <div class="signals-grid">
        `;
        
        indicatorSignals.forEach(sig => {
            html += `
                <div class="signal-card">
                    <div class="signal-card-header">
                        <span class="symbol-badge">${sig.symbol}</span>
                        <span class="signal-badge signal-buy">BUY</span>
                    </div>
                    <div class="signal-card-body">
                        <div class="signal-row">
                            <span class="label">Indicator:</span>
                            <span class="value">${sig.indicator}</span>
                        </div>
                        <div class="signal-row">
                            <span class="label">Value:</span>
                            <span class="value">${sig.value}</span>
                        </div>
                        <div class="signal-row">
                            <span class="label">Date:</span>
                            <span class="value">${new Date(sig.date).toLocaleDateString()}</span>
                        </div>
                    </div>
                    <div class="signal-card-footer">
                        <a href="/symbol/${encodeURIComponent(sig.symbol)}" class="view-link">View Details →</a>
                    </div>
                </div>
            `;
        });
        
        html += `
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

async function loadDashboardData() {
    try {
        const startTime = performance.now();
        console.log('🚀 [DASHBOARD] Starting data load...');
        
        document.body.classList.add('no-animation');
        
        // Parallel API calls - fetch fresh data
        const fetchStartTime = performance.now();
        console.log('📡 [FETCH] Starting parallel API calls...');
        
        const [summaryData, signalsData, indicatorsData] = await Promise.all([
            fetch(API.summary).then(r => r.json()),
            fetch(API.signals).then(r => r.json()),
            fetch(API.indicators).then(r => r.json())
        ]);
        
        const fetchEndTime = performance.now();
        console.log(`✅ [FETCH] API calls completed in ${(fetchEndTime - fetchStartTime).toFixed(2)}ms`);
        console.log(`📊 [DATA] Summary: ${summaryData.buy} BUY signals, ${summaryData.sell} SELL signals`);
        console.log(`📊 [DATA] Total symbols: ${summaryData.total_symbols}`);
        console.log(`📊 [DATA] Indicators: ${indicatorsData.length} unique indicators`);
        
        updateSummaryUI(summaryData, signalsData.length, indicatorsData.length);
        
        // Load latest day signals
        loadLatestSignals();
        
        setTimeout(() => {
            document.body.classList.remove('no-animation');
        }, 100);
        
        const totalTime = performance.now() - startTime;
        console.log(`⏱️  [TOTAL] Dashboard loaded in ${totalTime.toFixed(2)}ms`);
        console.log('═══════════════════════════════════════════════════════');
        
    } catch (error) {
        showError('Failed to load dashboard: ' + error.message);
        console.error('❌ [ERROR]', error);
    }
}

function updateSummaryUI(data, totalSignals, indicatorsCount) {
    const totalSymbols = document.getElementById('totalSymbols');
    const buyCount = document.getElementById('buyCount');
    const totalIndicators = document.getElementById('totalIndicators');
    const latestDate = document.getElementById('latestDate');
    
    if (totalSymbols) totalSymbols.textContent = data.total_symbols || 0;
    if (buyCount) buyCount.textContent = data.buy || 0;
    if (totalIndicators) totalIndicators.textContent = indicatorsCount || 0;
    if (latestDate) latestDate.textContent = `Latest Date: ${new Date(data.date).toLocaleDateString()}`;
}

async function loadSummaryInfo() {
    try {
        const response = await fetch(API.summary);
        const data = await response.json();
        
        const latestDate = document.getElementById('latestDate');
        if (latestDate) {
            latestDate.textContent = `Latest Date: ${new Date(data.date).toLocaleDateString()}`;
        }
    } catch (error) {
        console.error('Failed to load summary:', error);
    }
}

// ============================================
// PAGINATION - Display Signals by Page (NOT USED)
// ============================================
// Removed - using only tomorrow's signals now

// ============================================
// SYMBOL SEARCH
// ============================================
let searchTimeout;
const symbolSearch = document.getElementById('symbolSearch');
const searchResults = document.getElementById('searchResults');

if (symbolSearch) {
    symbolSearch.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        const query = this.value.trim();
        
        if (query.length < 1) {
            searchResults.classList.remove('show');
            return;
        }

        searchTimeout = setTimeout(async () => {
            try {
                const response = await fetch(`${API.symbols}?q=${encodeURIComponent(query)}`);
                const symbols = await response.json();
                
                searchResults.innerHTML = '';
                
                if (symbols.length === 0) {
                    searchResults.innerHTML = '<div class="search-item">No symbols found</div>';
                } else {
                    const fragment = document.createDocumentFragment();
                    symbols.slice(0, 10).forEach(symbol => {
                        const item = document.createElement('div');
                        item.className = 'search-item';
                        item.textContent = symbol;
                        item.addEventListener('click', () => {
                            window.location.href = `/symbol/${encodeURIComponent(symbol)}`;
                        });
                        fragment.appendChild(item);
                    });
                    searchResults.appendChild(fragment);
                }
                
                searchResults.classList.add('show');
            } catch (error) {
                showError('Failed to search symbols: ' + error.message);
            }
        }, 300);
    });

    document.addEventListener('click', function(event) {
        if (!symbolSearch.contains(event.target) && !searchResults.contains(event.target)) {
            searchResults.classList.remove('show');
        }
    });
}

// ============================================
// SYMBOL PAGE FUNCTIONS
// ============================================
async function loadSymbolIndicators() {
    if (typeof SYMBOL === 'undefined') return;
    
    try {
        const url = API.symbolIndicators(SYMBOL);
        const allResponse = await fetch(url);
        
        if (!allResponse.ok) {
            throw new Error(`HTTP error! status: ${allResponse.status}`);
        }
        
        const allIndicators = await allResponse.json();
        
        // Remove duplicates
        const uniqueIndicatorsMap = new Map();
        allIndicators.forEach(indicator => {
            const key = `${indicator.date}-${indicator.indicator}`;
            if (!uniqueIndicatorsMap.has(key)) {
                uniqueIndicatorsMap.set(key, indicator);
            }
        });
        
        const uniqueIndicators = Array.from(uniqueIndicatorsMap.values());
        
        const signalFilter = document.getElementById('signalFilter');
        const signal = signalFilter ? signalFilter.value : '';
        
        let filteredIndicators = uniqueIndicators;
        if (signal === 'NULL') {
            filteredIndicators = uniqueIndicators.filter(i => !i.signal);
        } else if (signal) {
            filteredIndicators = uniqueIndicators.filter(i => i.signal === signal);
        }
        
        const tbody = document.getElementById('indicatorsTableBody');
        if (!tbody) return;
        
        if (filteredIndicators.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="">No indicators found</td></tr>';
            return;
        }
        
        const fragment = document.createDocumentFragment();
        filteredIndicators.forEach(indicator => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${formatDate(indicator.date)}</td>
                <td>${indicator.indicator}</td>
                <td>${indicator.value !== null ? indicator.value.toFixed(2) : '-'}</td>
                <td>${formatSignal(indicator.signal)}</td>
            `;
            fragment.appendChild(row);
        });
        
        tbody.innerHTML = '';
        tbody.appendChild(fragment);
        
    } catch (error) {
        showError('Failed to load symbol indicators: ' + error.message);
    }
}

async function analyzeSymbol() {
    if (typeof SYMBOL === 'undefined') return;

    const indicator = document.getElementById('analysisIndicator').value;
    const target = document.getElementById('analysisTarget').value;
    const days = parseInt(document.getElementById('analysisDays').value);

    if (!indicator || !target || !days) {
        showNotification('Please fill in all fields: Indicator, Target %, and Days', 'warning');
        return;
    }

    // Validate days >= 1
    if (days < 1) {
        showNotification('⚠️ Days must be at least 1. DAYS=0 is diagnostic only, not suitable for live trading.', 'warning');
        return;
    }

    try {
        // Show loading state
        document.getElementById('analysisLoading').style.display = 'block';
        document.getElementById('analysisResult').style.display = 'none';
        
        // Disable analyze button
        const analyzeBtn = document.getElementById('analyzeBtn');
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'ANALYZING...';

        console.log(`📊 Analyzing ${SYMBOL} - ${indicator} with target=${target}%, days=${days}`);

        const url =
            `/api/analyze?symbol=${encodeURIComponent(SYMBOL)}` +
            `&indicator=${encodeURIComponent(indicator)}` +
            `&target=${target}&days=${days}`;

        const res = await fetch(url);
        const data = await res.json();

        if (!res.ok || data.error) {
            throw new Error(data.error || 'Analysis failed');
        }

        // Hide loading, show results
        document.getElementById('analysisLoading').style.display = 'none';
        document.getElementById('analysisResult').style.display = 'block';

        // ---- Summary ----
        document.getElementById('aTotal').textContent = data.totalSignals;
        document.getElementById('aCompleted').textContent = data.completedTrades;
        document.getElementById('aOpen').textContent = data.openTrades;
        document.getElementById('aSuccess').textContent = data.successful;
        document.getElementById('aRate').textContent = data.successRate;
        
        // Update profit/loss if elements exist
        if (document.getElementById('aAvgProfit')) {
            document.getElementById('aAvgProfit').textContent = data.avgMaxProfit !== undefined ? data.avgMaxProfit : '-';
        }
        if (document.getElementById('aAvgLoss')) {
            document.getElementById('aAvgLoss').textContent = data.avgMaxLoss !== undefined ? data.avgMaxLoss : '-';
        }

        // ---- Decision badge ----
        const badge = document.getElementById('decisionBadge');
        badge.className = 'decision-badge';

        if (data.successRate >= 70) {
            badge.textContent = 'HIGH HISTORICAL FOLLOW-THROUGH';
            badge.classList.add('decision-strong');
        } else if (data.successRate >= 50) {
            badge.textContent = 'MODERATE FOLLOW-THROUGH';
            badge.classList.add('decision-medium');
        } else {
            badge.textContent = 'WEAK FOLLOW-THROUGH';
            badge.classList.add('decision-weak');
        }

        document.getElementById('recommendationText').textContent =
            `Out of ${data.completedTrades} completed signals, ` +
            `${data.successful} reached +${data.targetPct}% within ` +
            `${data.days} trading days.`;

        // ---- Details table (SORTED BY DATE DESCENDING - LATEST FIRST) ----
        const tbody = document.getElementById('detailsTableBody');
        tbody.innerHTML = '';

        if (!data.details || data.details.length === 0) {
            tbody.innerHTML =
                `<tr><td colspan="6" class="center">No historical BUY signals found for this indicator</td></tr>`;
        } else {
            // Sort by date descending (latest first)
            const sortedDetails = [...data.details].sort((a, b) => {
                return new Date(b.buyDate) - new Date(a.buyDate);
            });

            const fragment = document.createDocumentFragment();

            sortedDetails.forEach(d => {
                const tr = document.createElement('tr');

                let resultClass = '';
                if (d.result === 'SUCCESS') resultClass = 'target-hit-yes';
                else if (d.result === 'FAIL') resultClass = 'target-hit-no';
                else resultClass = 'trade-open';

                tr.innerHTML = `
                    <td>${new Date(d.buyDate).toLocaleDateString()}</td>
                    <td>₹${d.buyPrice}</td>
                    <td>₹${d.targetPrice}</td>
                    <td>${d.maxPriceReached ? '₹' + d.maxPriceReached : '-'}</td>
                    <td>${d.daysChecked}</td>
                    <td class="${resultClass}">${d.result}</td>
                `;
                fragment.appendChild(tr);
            });

            tbody.appendChild(fragment);
        }

        showNotification(
            `✅ Analysis complete: ${data.successRate}% success rate`,
            'success'
        );

        // Auto-load chart for the selected indicator
        autoLoadChartForIndicator(indicator);

        // Re-enable button
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'ANALYZE';

    } catch (err) {
        document.getElementById('analysisLoading').style.display = 'none';
        showError('Analysis failed: ' + err.message);
        
        // Re-enable button
        const analyzeBtn = document.getElementById('analyzeBtn');
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'ANALYZE';
    }
}

// Auto-load chart based on indicator type
function autoLoadChartForIndicator(indicator) {
    if (!indicator) return;
    
    console.log(`📊 Auto-loading chart for indicator: ${indicator}`);
    
    // Determine indicator type and set appropriate dropdown
    if (indicator.includes('SMA')) {
        const smaSelect = document.getElementById('smaSelect');
        if (smaSelect) {
            smaSelect.value = indicator;
            console.log(`✅ Set SMA dropdown to: ${indicator}`);
        }
    } else if (indicator.includes('RSI')) {
        const rsiSelect = document.getElementById('rsiSelect');
        if (rsiSelect) {
            rsiSelect.value = indicator;
            console.log(`✅ Set RSI dropdown to: ${indicator}`);
        }
    } else if (indicator.includes('BB')) {
        // For BB indicators, extract the period and set the dropdown to the Middle band
        // But store the actual indicator (Upper/Middle/Lower) for the API
        const bbMatch = indicator.match(/BB(\d+)_(Upper|Middle|Lower)/);
        if (bbMatch) {
            const period = bbMatch[1];
            const type = bbMatch[2];
            const bbSelect = document.getElementById('bbSelect');
            if (bbSelect) {
                // Set dropdown to the Middle band (that's what's in the dropdown)
                bbSelect.value = `BB${period}_Middle`;
                // Store the actual analyzed indicator as a data attribute
                bbSelect.setAttribute('data-analyzed-indicator', indicator);
                console.log(`✅ Set BB dropdown to: BB${period}_Middle (analyzed: ${indicator})`);
            }
        }
    } else if (indicator === 'Short' || indicator === 'Long' || indicator === 'Standard') {
        const macdSelect = document.getElementById('macdSelect');
        if (macdSelect) {
            macdSelect.value = indicator;
            console.log(`✅ Set MACD dropdown to: ${indicator}`);
        }
    } else if (indicator.includes('STOCH')) {
        const stochSelect = document.getElementById('stochSelect');
        if (stochSelect) {
            stochSelect.value = indicator;
            console.log(`✅ Set Stochastic dropdown to: ${indicator}`);
        }
    }
    
    // Update the chart with the selected indicator
    setTimeout(() => {
        updateChart();
    }, 300);
}

async function loadChartIndicators() {
    if (typeof SYMBOL === 'undefined') return;
    
    try {
        console.log('📊 [CHART] Loading chart indicators...');
        const response = await fetch(API.indicators);
        const indicators = await response.json();
        
        console.log(`📊 [CHART] Received ${indicators.length} indicators:`, indicators);
        
        const smaSelect = document.getElementById('smaSelect');
        const rsiSelect = document.getElementById('rsiSelect');
        const bbSelect = document.getElementById('bbSelect');
        const macdSelect = document.getElementById('macdSelect');
        const stochSelect = document.getElementById('stochSelect');
        
        if (smaSelect) {
            const smaIndicators = indicators.filter(i => i.includes('SMA'));
            console.log(`📊 [CHART] SMA indicators:`, smaIndicators);
            const fragment = document.createDocumentFragment();
            smaIndicators.forEach(indicator => {
                const option = document.createElement('option');
                option.value = indicator;
                option.textContent = indicator;
                fragment.appendChild(option);
            });
            smaSelect.appendChild(fragment);
            console.log(`✅ [CHART] Added ${smaIndicators.length} SMA options`);
        }
        
        if (rsiSelect) {
            const rsiIndicators = indicators.filter(i => i.includes('RSI'));
            console.log(`📊 [CHART] RSI indicators:`, rsiIndicators);
            const fragment = document.createDocumentFragment();
            rsiIndicators.forEach(indicator => {
                const option = document.createElement('option');
                option.value = indicator;
                option.textContent = indicator;
                fragment.appendChild(option);
            });
            rsiSelect.appendChild(fragment);
            console.log(`✅ [CHART] Added ${rsiIndicators.length} RSI options`);
        }
        
        if (bbSelect) {
            const fragment = document.createDocumentFragment();
            // Get all BB indicators (Upper, Middle, Lower)
            const bbIndicators = indicators.filter(i => i.includes('BB'));
            console.log(`📊 [CHART] BB indicators:`, bbIndicators);
            
            // Group by period
            const bbByPeriod = {};
            bbIndicators.forEach(ind => {
                const match = ind.match(/BB(\d+)_(Upper|Middle|Lower)/);
                if (match) {
                    const period = match[1];
                    const type = match[2];
                    if (!bbByPeriod[period]) {
                        bbByPeriod[period] = {};
                    }
                    bbByPeriod[period][type] = ind;
                }
            });
            
            console.log(`📊 [CHART] BB by period:`, bbByPeriod);
            
            // Add options for each period (use Lower band for BUY signals)
            Object.keys(bbByPeriod).sort((a, b) => parseInt(a) - parseInt(b)).forEach(period => {
                // Prefer Lower band (where BUY signals typically are), fallback to Middle
                const bbValue = bbByPeriod[period].Lower || bbByPeriod[period].Middle;
                if (bbValue) {
                    const option = document.createElement('option');
                    option.value = bbValue;
                    option.textContent = `BB${period}`;
                    fragment.appendChild(option);
                }
            });
            
            bbSelect.appendChild(fragment);
            console.log(`✅ [CHART] Added ${Object.keys(bbByPeriod).length} BB options`);
        }
        
        if (macdSelect) {
            const fragment = document.createDocumentFragment();
            ['Short', 'Long', 'Standard'].forEach(indicator => {
                const option = document.createElement('option');
                option.value = indicator;
                option.textContent = `MACD ${indicator}`;
                fragment.appendChild(option);
            });
            macdSelect.appendChild(fragment);
            console.log(`✅ [CHART] Added 3 MACD options`);
        }
        
        if (stochSelect) {
            const stochIndicators = indicators.filter(i => i.includes('STOCH'));
            console.log(`📊 [CHART] Stochastic indicators:`, stochIndicators);
            const fragment = document.createDocumentFragment();
            stochIndicators.forEach(indicator => {
                const option = document.createElement('option');
                option.value = indicator;
                option.textContent = indicator;
                fragment.appendChild(option);
            });
            stochSelect.appendChild(fragment);
            console.log(`✅ [CHART] Added ${stochIndicators.length} Stochastic options`);
        }
        
        console.log('✅ [CHART] All chart indicators loaded successfully');
        
        // Also load analysis indicators
        loadAnalysisIndicators(indicators);
        
    } catch (error) {
        console.error('❌ [CHART] Failed to load chart indicators:', error);
        showError('Failed to load chart indicators: ' + error.message);
    }
}

async function loadAnalysisIndicators(indicators = null) {
    if (typeof SYMBOL === 'undefined') return;
    
    try {
        console.log('📊 [ANALYSIS] Loading analysis indicators...');
        
        // If indicators not provided, fetch them
        if (!indicators) {
            const response = await fetch(API.indicators);
            indicators = await response.json();
        }
        
        console.log(`📊 [ANALYSIS] Received ${indicators.length} indicators for analysis dropdown`);
        
        const analysisSelect = document.getElementById('analysisIndicator');
        if (!analysisSelect) {
            console.log('⚠️ [ANALYSIS] Analysis indicator dropdown not found');
            return;
        }
        
        // Clear existing options except the first one (placeholder)
        while (analysisSelect.options.length > 1) {
            analysisSelect.remove(1);
        }
        
        // Filter indicators by type
        const smaIndicators = indicators.filter(i => i.includes('SMA')).sort();
        const rsiIndicators = indicators.filter(i => i.includes('RSI')).sort();
        const bbIndicators = indicators.filter(i => i.includes('BB')).sort();
        const macdIndicators = indicators.filter(i => ['Short', 'Long', 'Standard'].includes(i));
        const stochIndicators = indicators.filter(i => i.includes('STOCH')).sort();
        
        // Add SMA group
        if (smaIndicators.length > 0) {
            const smaGroup = document.createElement('optgroup');
            smaGroup.label = 'SMA Indicators';
            smaIndicators.forEach(indicator => {
                const option = document.createElement('option');
                option.value = indicator;
                option.textContent = indicator;
                smaGroup.appendChild(option);
            });
            analysisSelect.appendChild(smaGroup);
        }
        
        // Add RSI group
        if (rsiIndicators.length > 0) {
            const rsiGroup = document.createElement('optgroup');
            rsiGroup.label = 'RSI Indicators';
            rsiIndicators.forEach(indicator => {
                const option = document.createElement('option');
                option.value = indicator;
                option.textContent = indicator;
                rsiGroup.appendChild(option);
            });
            analysisSelect.appendChild(rsiGroup);
        }
        
        // Add BB group
        if (bbIndicators.length > 0) {
            const bbGroup = document.createElement('optgroup');
            bbGroup.label = 'Bollinger Bands';
            bbIndicators.forEach(indicator => {
                const option = document.createElement('option');
                option.value = indicator;
                // Format display name (e.g., "BB10_Upper" -> "BB10 Upper")
                const displayName = indicator.replace('_', ' ');
                option.textContent = displayName;
                bbGroup.appendChild(option);
            });
            analysisSelect.appendChild(bbGroup);
        }
        
        // Add MACD group
        if (macdIndicators.length > 0) {
            const macdGroup = document.createElement('optgroup');
            macdGroup.label = 'MACD Indicators';
            macdIndicators.forEach(indicator => {
                const option = document.createElement('option');
                option.value = indicator;
                option.textContent = `MACD ${indicator}`;
                macdGroup.appendChild(option);
            });
            analysisSelect.appendChild(macdGroup);
        }
        
        // Add Stochastic group
        if (stochIndicators.length > 0) {
            const stochGroup = document.createElement('optgroup');
            stochGroup.label = 'Stochastic Indicators';
            stochIndicators.forEach(indicator => {
                const option = document.createElement('option');
                option.value = indicator;
                option.textContent = indicator;
                stochGroup.appendChild(option);
            });
            analysisSelect.appendChild(stochGroup);
        }
        
        console.log(`✅ [ANALYSIS] Populated analysis dropdown with ${indicators.length} indicators`);
        console.log(`   - SMA: ${smaIndicators.length}, RSI: ${rsiIndicators.length}, BB: ${bbIndicators.length}, MACD: ${macdIndicators.length}, STOCH: ${stochIndicators.length}`);
        
    } catch (error) {
        console.error('❌ [ANALYSIS] Failed to load analysis indicators:', error);
    }
}

function updateAnalysisPeriods() {
    // No longer needed - all indicators are now directly selectable
}

let priceChart = null;
let bbChart = null;
let rsiChart = null;
let macdChart = null;
let stochChart = null;

async function updateChart() {
    if (typeof SYMBOL === 'undefined') return;
    
    try {
        const smaSelect = document.getElementById('smaSelect');
        const rsiSelect = document.getElementById('rsiSelect');
        const bbSelect = document.getElementById('bbSelect');
        const macdSelect = document.getElementById('macdSelect');
        const stochSelect = document.getElementById('stochSelect');
        
        const sma = smaSelect ? smaSelect.value : '';
        const rsi = rsiSelect ? rsiSelect.value : '';
        let bb = bbSelect ? bbSelect.value : '';
        const macd = macdSelect ? macdSelect.value : '';
        const stoch = stochSelect ? stochSelect.value : '';
        
        // Check if BB has an analyzed indicator stored (from autoLoadChartForIndicator)
        if (bbSelect && bbSelect.hasAttribute('data-analyzed-indicator')) {
            bb = bbSelect.getAttribute('data-analyzed-indicator');
            console.log(`📊 [CHART] Using analyzed BB indicator: ${bb}`);
        }
        
        // Build URL with query parameters for selected indicators
        let url = API.symbolChart(SYMBOL);
        const params = new URLSearchParams();
        if (sma) params.append('sma', sma);
        if (rsi) params.append('rsi', rsi);
        if (bb) params.append('bb', bb);
        if (macd) params.append('macd', macd);
        if (stoch) params.append('stoch', stoch);
        
        if (params.toString()) {
            url += '?' + params.toString();
        }
        
        console.log(`📊 [CHART] Loading chart data from: ${url}`);
        console.log(`📊 [CHART] Selected indicators - SMA: ${sma}, RSI: ${rsi}, BB: ${bb}, MACD: ${macd}, STOCH: ${stoch}`);
        
        const response = await fetch(url);
        const data = await response.json();
        
        console.log(`📊 [CHART] Received ${data.length} data points`);
        if (data.length > 0) {
            console.log(`📊 [CHART] Sample data point:`, data[0]);
        }
        
        if (data.length === 0) {
            showError('No chart data available');
            return;
        }
        
        // ===== CHART 1: PRICE + SMA =====
        const priceContainer = document.getElementById('priceChart');
        if (priceContainer && typeof LightweightCharts !== 'undefined') {
            if (priceChart) {
                priceChart.remove();
            }
            
            priceChart = LightweightCharts.createChart(priceContainer, {
                width: priceContainer.clientWidth,
                height: 400,
                layout: {
                    backgroundColor: '#ffffff',
                    textColor: '#333333',
                    fontSize: 12,
                },
                grid: {
                    vertLines: { color: '#f0f0f0', style: 1, visible: true },
                    horzLines: { color: '#f0f0f0', style: 1, visible: true },
                },
                timeScale: {
                    borderColor: '#cccccc',
                    timeVisible: true,
                    secondsVisible: false,
                },
                rightPriceScale: {
                    borderColor: '#cccccc',
                    scaleMargins: { top: 0.1, bottom: 0.1 },
                },
                crosshair: { mode: 1 },
            });
            
            // Price line
            const priceSeries = priceChart.addLineSeries({
                color: '#2563eb',
                lineWidth: 2,
                title: `${SYMBOL} Price`,
            });
            
            const priceData = data.map(d => ({
                time: d.date,
                value: d.price
            }));
            
            priceSeries.setData(priceData);
            
            // SMA line if selected
            if (sma && data.some(d => d.sma !== null)) {
                const smaSeries = priceChart.addLineSeries({
                    color: '#10b981',
                    lineWidth: 2,
                    title: sma,
                });
                
                const smaData = data
                    .filter(d => d.sma !== null)
                    .map(d => ({
                        time: d.date,
                        value: d.sma
                    }));
                
                smaSeries.setData(smaData);
            }
            
            // Add BUY signal markers for SMA
            if (sma && data.some(d => d.sma_signal === 'BUY')) {
                const buySignals = data
                    .filter(d => d.sma_signal === 'BUY' && d.price !== null)
                    .map(d => ({
                        time: d.date,
                        position: 'belowBar',
                        color: '#10b981',
                        shape: 'arrowUp',
                        text: 'BUY',
                        size: 2
                    }));
                
                if (buySignals.length > 0) {
                    priceSeries.setMarkers(buySignals);
                    console.log(`📊 [SMA] Added ${buySignals.length} BUY signal markers for ${sma}`);
                }
            }
            
            priceChart.timeScale().fitContent();
        }
        
        // ===== CHART 2: BOLLINGER BANDS =====
        const bbContainer = document.getElementById('bbChart');
        if (bbContainer && typeof LightweightCharts !== 'undefined') {
            if (bbChart) {
                bbChart.remove();
            }
            
            if (bb && data.some(d => d.bb_upper !== null)) {
                bbContainer.classList.remove('hidden');
                
                bbChart = LightweightCharts.createChart(bbContainer, {
                    width: bbContainer.clientWidth,
                    height: 300,
                    layout: {
                        backgroundColor: '#ffffff',
                        textColor: '#333333',
                        fontSize: 12,
                    },
                    grid: {
                        vertLines: { color: '#f0f0f0', style: 1, visible: true },
                        horzLines: { color: '#f0f0f0', style: 1, visible: true },
                    },
                    timeScale: {
                        borderColor: '#cccccc',
                        timeVisible: true,
                        secondsVisible: false,
                    },
                    rightPriceScale: {
                        borderColor: '#cccccc',
                        scaleMargins: { top: 0.1, bottom: 0.1 },
                    },
                    crosshair: { mode: 1 },
                });
                
                // Price line
                const bbPriceSeries = bbChart.addLineSeries({
                    color: '#2563eb',
                    lineWidth: 2,
                    title: `${SYMBOL} Price`,
                });
                
                const bbPriceData = data.map(d => ({
                    time: d.date,
                    value: d.price
                }));
                
                bbPriceSeries.setData(bbPriceData);
                
                // Upper band
                const bbUpperSeries = bbChart.addLineSeries({
                    color: '#ef4444',
                    lineWidth: 1,
                    lineStyle: 2,
                    title: `${bb} Upper`,
                });
                
                const bbUpperData = data
                    .filter(d => d.bb_upper !== null)
                    .map(d => ({
                        time: d.date,
                        value: d.bb_upper
                    }));
                
                bbUpperSeries.setData(bbUpperData);
                
                // Middle band
                const bbMiddleSeries = bbChart.addLineSeries({
                    color: '#f59e0b',
                    lineWidth: 2,
                    title: `${bb} Middle`,
                });
                
                const bbMiddleData = data
                    .filter(d => d.bb_middle !== null)
                    .map(d => ({
                        time: d.date,
                        value: d.bb_middle
                    }));
                
                bbMiddleSeries.setData(bbMiddleData);
                
                // Lower band
                const bbLowerSeries = bbChart.addLineSeries({
                    color: '#10b981',
                    lineWidth: 1,
                    lineStyle: 2,
                    title: `${bb} Lower`,
                });
                
                const bbLowerData = data
                    .filter(d => d.bb_lower !== null)
                    .map(d => ({
                        time: d.date,
                        value: d.bb_lower
                    }));
                
                bbLowerSeries.setData(bbLowerData);
                
                // Add BUY signal markers for BB
                const bbBuySignals = data
                    .filter(d => d.bb_signal === 'BUY' && d.price !== null)
                    .map(d => ({
                        time: d.date,
                        position: 'belowBar',
                        color: '#10b981',
                        shape: 'arrowUp',
                        text: 'BUY',
                        size: 2
                    }));
                
                console.log(`📊 [BB] Found ${bbBuySignals.length} BUY signals`);
                
                if (bbBuySignals.length > 0) {
                    bbPriceSeries.setMarkers(bbBuySignals);
                    console.log(`📊 [BB] Added ${bbBuySignals.length} BUY signal markers`);
                }
                
                bbChart.timeScale().fitContent();
            } else {
                bbContainer.classList.add('hidden');
                if (bbChart) {
                    bbChart.remove();
                    bbChart = null;
                }
            }
        }
        
        // ===== CHART 3: RSI =====
        const rsiContainer = document.getElementById('rsiChart');
        if (rsiContainer && typeof LightweightCharts !== 'undefined') {
            if (rsiChart) {
                rsiChart.remove();
            }
            
            if (rsi && data.some(d => d.rsi !== null)) {
                rsiContainer.classList.remove('hidden');
                
                rsiChart = LightweightCharts.createChart(rsiContainer, {
                    width: rsiContainer.clientWidth,
                    height: 250,
                    layout: {
                        backgroundColor: '#fafafa',
                        textColor: '#333333',
                        fontSize: 11,
                    },
                    grid: {
                        vertLines: { color: '#f0f0f0', style: 1, visible: true },
                        horzLines: { color: '#f0f0f0', style: 1, visible: true },
                    },
                    timeScale: {
                        borderColor: '#cccccc',
                        timeVisible: false,
                        secondsVisible: false,
                    },
                    rightPriceScale: {
                        borderColor: '#cccccc',
                        scaleMargins: { top: 0.1, bottom: 0.1 },
                        entireTextOnly: true,
                    },
                    crosshair: { mode: 1 },
                });
                
                const rsiSeries = rsiChart.addLineSeries({
                    color: '#ef4444',
                    lineWidth: 2,
                    title: rsi,
                });
                
                const rsiData = data
                    .filter(d => d.rsi !== null)
                    .map(d => ({
                        time: d.date,
                        value: d.rsi
                    }));
                
                rsiSeries.setData(rsiData);
                
                if (rsiData.length > 0) {
                    const overBoughtSeries = rsiChart.addLineSeries({
                        color: '#dc2626',
                        lineWidth: 1,
                        lineStyle: 2,
                        title: 'Overbought (70)',
                    });
                    
                    const overSoldSeries = rsiChart.addLineSeries({
                        color: '#16a34a',
                        lineWidth: 1,
                        lineStyle: 2,
                        title: 'Oversold (30)',
                    });
                    
                    const firstTime = rsiData[0].time;
                    const lastTime = rsiData[rsiData.length - 1].time;
                    
                    overBoughtSeries.setData([
                        { time: firstTime, value: 70 },
                        { time: lastTime, value: 70 }
                    ]);
                    
                    overSoldSeries.setData([
                        { time: firstTime, value: 30 },
                        { time: lastTime, value: 30 }
                    ]);
                }
                
                // Add BUY signal markers for RSI
                const rsiBuySignals = data
                    .filter(d => d.rsi_signal === 'BUY' && d.rsi !== null)
                    .map(d => ({
                        time: d.date,
                        position: 'belowBar',
                        color: '#10b981',
                        shape: 'arrowUp',
                        text: 'BUY',
                        size: 2
                    }));
                
                if (rsiBuySignals.length > 0) {
                    rsiSeries.setMarkers(rsiBuySignals);
                    console.log(`📊 [RSI] Added ${rsiBuySignals.length} BUY signal markers`);
                }
                
                rsiChart.timeScale().fitContent();
            } else {
                rsiContainer.classList.add('hidden');
                if (rsiChart) {
                    rsiChart.remove();
                    rsiChart = null;
                }
            }
        }
        
        // ===== CHART 4: MACD =====
        const macdContainer = document.getElementById('macdChart');
        
        if (macdContainer && typeof LightweightCharts !== 'undefined') {
            if (macdChart) {
                macdChart.remove();
            }
            
            console.log(`📊 [MACD] Selected MACD: ${macd}`);
            console.log(`📊 [MACD] Has MACD data:`, data.some(d => d.macd_line !== null || d.macd_signal !== null || d.macd_histogram !== null));
            
            if (macd && data.some(d => d.macd_line !== null || d.macd_signal !== null || d.macd_histogram !== null)) {
                macdContainer.classList.remove('hidden');
                
                console.log(`📊 [MACD] Creating MACD chart...`);
                
                macdChart = LightweightCharts.createChart(macdContainer, {
                    width: macdContainer.clientWidth,
                    height: 300,
                    layout: {
                        backgroundColor: '#ffffff',
                        textColor: '#333333',
                        fontSize: 11,
                    },
                    grid: {
                        vertLines: { color: '#f0f0f0', style: 1, visible: true },
                        horzLines: { color: '#f0f0f0', style: 1, visible: true },
                    },
                    timeScale: {
                        borderColor: '#cccccc',
                        timeVisible: true,
                        secondsVisible: false,
                    },
                    rightPriceScale: {
                        borderColor: '#cccccc',
                        scaleMargins: { top: 0.1, bottom: 0.1 },
                    },
                    crosshair: { 
                        mode: 1,
                        vertLine: {
                            width: 1,
                            color: '#758696',
                            style: 3,
                        },
                        horzLine: {
                            width: 1,
                            color: '#758696',
                            style: 3,
                        },
                    },
                });
                
                // MACD Histogram (draw first, so it's in the background)
                const macdHistData = data
                    .filter(d => d.macd_histogram !== null)
                    .map(d => ({
                        time: d.date,
                        value: d.macd_histogram,
                        color: d.macd_histogram >= 0 ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
                    }));
                
                console.log(`📊 [MACD] MACD Histogram data points: ${macdHistData.length}`);
                
                let macdHistSeries = null;
                if (macdHistData.length > 0) {
                    macdHistSeries = macdChart.addHistogramSeries({
                        priceFormat: {
                            type: 'price',
                            precision: 4,
                            minMove: 0.0001,
                        },
                        title: 'Histogram',
                    });
                    macdHistSeries.setData(macdHistData);
                }
                
                // MACD Line (blue)
                const macdLineData = data
                    .filter(d => d.macd_line !== null)
                    .map(d => ({
                        time: d.date,
                        value: d.macd_line
                    }));
                
                console.log(`📊 [MACD] MACD Line data points: ${macdLineData.length}`);
                
                let macdLineSeries = null;
                if (macdLineData.length > 0) {
                    macdLineSeries = macdChart.addLineSeries({
                        color: '#2962FF',
                        lineWidth: 2,
                        title: 'MACD',
                        priceFormat: {
                            type: 'price',
                            precision: 4,
                            minMove: 0.0001,
                        },
                    });
                    macdLineSeries.setData(macdLineData);
                }
                
                // MACD Signal Line (orange/red)
                const macdSignalData = data
                    .filter(d => d.macd_signal !== null)
                    .map(d => ({
                        time: d.date,
                        value: d.macd_signal
                    }));
                
                console.log(`📊 [MACD] MACD Signal data points: ${macdSignalData.length}`);
                
                if (macdSignalData.length > 0) {
                    const macdSignalSeries = macdChart.addLineSeries({
                        color: '#FF6D00',
                        lineWidth: 2,
                        title: 'Signal',
                        priceFormat: {
                            type: 'price',
                            precision: 4,
                            minMove: 0.0001,
                        },
                    });
                    macdSignalSeries.setData(macdSignalData);
                }
                
                // Add zero line for reference
                if (macdLineData.length > 0) {
                    const zeroLineSeries = macdChart.addLineSeries({
                        color: '#787B86',
                        lineWidth: 1,
                        lineStyle: 2, // dashed
                        title: 'Zero',
                        priceLineVisible: false,
                        lastValueVisible: false,
                    });
                    
                    const firstTime = macdLineData[0].time;
                    const lastTime = macdLineData[macdLineData.length - 1].time;
                    
                    zeroLineSeries.setData([
                        { time: firstTime, value: 0 },
                        { time: lastTime, value: 0 }
                    ]);
                }
                
                // Add BUY signal markers for MACD
                const macdBuySignals = data
                    .filter(d => d.macd_signal_flag === 'BUY' && d.macd_line !== null)
                    .map(d => ({
                        time: d.date,
                        position: 'belowBar',
                        color: '#10b981',
                        shape: 'arrowUp',
                        text: 'BUY',
                        size: 2
                    }));
                
                console.log(`📊 [MACD] Found ${macdBuySignals.length} BUY signals`);
                
                if (macdBuySignals.length > 0 && macdLineSeries) {
                    macdLineSeries.setMarkers(macdBuySignals);
                    console.log(`📊 [MACD] Added ${macdBuySignals.length} BUY signal markers`);
                }
                
                macdChart.timeScale().fitContent();
                console.log(`✅ [MACD] Chart created successfully`);
            } else {
                console.log(`📊 [MACD] Hiding MACD chart - no data or not selected`);
                macdContainer.classList.add('hidden');
                if (macdChart) {
                    macdChart.remove();
                    macdChart = null;
                }
            }
        }
        
        // ===== CHART 5: STOCHASTIC =====
        const stochContainer = document.getElementById('stochChart');
        
        if (stochContainer && typeof LightweightCharts !== 'undefined') {
            if (stochChart) {
                stochChart.remove();
            }
            
            if (stoch && data.some(d => d.stoch_k !== null)) {
                stochContainer.classList.remove('hidden');
                
                stochChart = LightweightCharts.createChart(stochContainer, {
                    width: stochContainer.clientWidth,
                    height: 250,
                    layout: {
                        backgroundColor: '#fafafa',
                        textColor: '#333333',
                        fontSize: 11,
                    },
                    grid: {
                        vertLines: { color: '#f0f0f0', style: 1, visible: true },
                        horzLines: { color: '#f0f0f0', style: 1, visible: true },
                    },
                    timeScale: {
                        borderColor: '#cccccc',
                        timeVisible: false,
                        secondsVisible: false,
                    },
                    rightPriceScale: {
                        borderColor: '#cccccc',
                        scaleMargins: { top: 0.1, bottom: 0.1 },
                    },
                    crosshair: { mode: 1 },
                });
                
                // %K Line
                const stochKSeries = stochChart.addLineSeries({
                    color: '#3b82f6',
                    lineWidth: 2,
                    title: '%K',
                });
                
                const stochKData = data
                    .filter(d => d.stoch_k !== null)
                    .map(d => ({
                        time: d.date,
                        value: d.stoch_k
                    }));
                
                if (stochKData.length > 0) {
                    stochKSeries.setData(stochKData);
                    
                    // %D Line
                    const stochDSeries = stochChart.addLineSeries({
                        color: '#ef4444',
                        lineWidth: 2,
                        title: '%D',
                    });
                    
                    const stochDData = data
                        .filter(d => d.stoch_d !== null)
                        .map(d => ({
                            time: d.date,
                            value: d.stoch_d
                        }));
                    
                    if (stochDData.length > 0) {
                        stochDSeries.setData(stochDData);
                    }
                    
                    // Overbought line (80)
                    const overboughtSeries = stochChart.addLineSeries({
                        color: '#dc2626',
                        lineWidth: 1,
                        lineStyle: 2,
                        title: 'Overbought (80)',
                    });
                    
                    // Oversold line (20)
                    const oversoldSeries = stochChart.addLineSeries({
                        color: '#16a34a',
                        lineWidth: 1,
                        lineStyle: 2,
                        title: 'Oversold (20)',
                    });
                    
                    const firstTime = stochKData[0].time;
                    const lastTime = stochKData[stochKData.length - 1].time;
                    
                    overboughtSeries.setData([
                        { time: firstTime, value: 80 },
                        { time: lastTime, value: 80 }
                    ]);
                    
                    oversoldSeries.setData([
                        { time: firstTime, value: 20 },
                        { time: lastTime, value: 20 }
                    ]);
                }
                
                // Add BUY signal markers for Stochastic
                const stochBuySignals = data
                    .filter(d => d.stoch_signal === 'BUY' && d.stoch_k !== null)
                    .map(d => ({
                        time: d.date,
                        position: 'belowBar',
                        color: '#10b981',
                        shape: 'arrowUp',
                        text: 'BUY',
                        size: 2
                    }));
                
                if (stochBuySignals.length > 0) {
                    stochKSeries.setMarkers(stochBuySignals);
                    console.log(`📊 [STOCH] Added ${stochBuySignals.length} BUY signal markers`);
                }
                
                stochChart.timeScale().fitContent();
            } else {
                stochContainer.classList.add('hidden');
                if (stochChart) {
                    stochChart.remove();
                    stochChart = null;
                }
            }
        }
        
    } catch (error) {
        showError('Failed to load chart data: ' + error.message);
    }
}

// ============================================
// DASHBOARD ANALYSIS FUNCTIONS
// ============================================
async function analyzeDashboard() {
    const target = document.getElementById('dashboardTarget').value;
    const days = document.getElementById('dashboardDays').value;

    if (!target || !days) {
        showNotification('Fill target and days', 'warning');
        return;
    }

    try {
        // Show loading state
        document.getElementById('loadingState').classList.remove('hidden');
        document.getElementById('resultsSection').classList.add('hidden');

        console.log(`📊 [DASHBOARD] Analyzing all BUY signals with target=${target}%, days=${days}`);
        const startTime = performance.now();

        // PROGRESSIVE LOADING: Load first 50 results immediately
        let allResults = [];
        let offset = 0;
        let hasMore = true;
        let isFirstBatch = true;
        
        while (hasMore) {
            const analyzeUrl = `/api/analyze-progressive?target=${target}&days=${days}&batch_size=50&offset=${offset}`;
            const response = await fetch(analyzeUrl);
            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Analysis failed');
            }

            // Add results from this batch
            if (data.results && data.results.length > 0) {
                allResults = allResults.concat(data.results);
                
                // Show first batch immediately
                if (isFirstBatch) {
                    console.log(`✅ [DASHBOARD] First ${data.batch_size} results loaded in ${data.processing_time_seconds}s`);
                    
                    // Show results section with first batch
                    document.getElementById('loadingState').classList.add('hidden');
                    document.getElementById('resultsSection').classList.remove('hidden');
                    
                    // Display first batch
                    displayDashboardResults(allResults, target, days, {
                        isPartial: true,
                        loaded: allResults.length,
                        total: data.total_signals
                    });
                    
                    isFirstBatch = false;
                } else {
                    // Update with accumulated results
                    console.log(`📊 [DASHBOARD] Loaded ${allResults.length}/${data.total_signals} results...`);
                    displayDashboardResults(allResults, target, days, {
                        isPartial: hasMore,
                        loaded: allResults.length,
                        total: data.total_signals
                    });
                }
            }

            // Check if there are more results
            hasMore = data.has_more;
            offset = data.next_offset || 0;
            
            // Break if no more results
            if (!hasMore) {
                break;
            }
        }

        const endTime = performance.now();
        console.log(`✅ [DASHBOARD] All ${allResults.length} results loaded in ${((endTime - startTime) / 1000).toFixed(2)}s`);

        // Final update with all results
        displayDashboardResults(allResults, target, days, {
            isPartial: false,
            loaded: allResults.length,
            total: allResults.length
        });

        // Calculate average success rate
        const avgSuccessRate = allResults.length > 0 
            ? (allResults.reduce((sum, r) => sum + r.successRate, 0) / allResults.length).toFixed(2)
            : 0;

        showNotification(`✅ Analyzed ${allResults.length} signals - Avg success: ${avgSuccessRate}%`, 'success');

    } catch (error) {
        document.getElementById('loadingState').classList.add('hidden');
        showError('Dashboard analysis failed: ' + error.message);
        console.error('❌ [ERROR]', error);
    }
}

function displaySkeletonLoaders(count = 6) {
    const container = document.getElementById('resultsContainer');
    
    let html = `
        <div class="results-progress-header">
            <span class="progress-count">
                📊 <strong>Loading</strong> results...
            </span>
            <span class="progress-indicator">
                <span class="loading-dots">
                    <span class="loading-dot"></span>
                    <span class="loading-dot"></span>
                    <span class="loading-dot"></span>
                </span>
            </span>
        </div>
        
        <table class="results-table skeleton-table">
            <thead>
                <tr>
                    <th class="col-no">No.</th>
                    <th class="col-symbol">Company Symbol</th>
                    <th class="col-indicator">Indicator</th>
                    <th class="col-total">Total Signals</th>
                    <th class="col-success">Success</th>
                    <th class="col-failure">Failure</th>
                    <th class="col-open">Open</th>
                    <th class="col-rate">Success %</th>
                    <th class="col-action">Action</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    for (let i = 0; i < count; i++) {
        html += `
            <tr class="skeleton-row">
                <td><div class="skeleton-line short"></div></td>
                <td><div class="skeleton-line"></div></td>
                <td><div class="skeleton-line"></div></td>
                <td><div class="skeleton-line short"></div></td>
                <td><div class="skeleton-line short"></div></td>
                <td><div class="skeleton-line short"></div></td>
                <td><div class="skeleton-line short"></div></td>
                <td><div class="skeleton-line short"></div></td>
                <td><div class="skeleton-line"></div></td>
            </tr>
        `;
    }
    
    html += `
            </tbody>
        </table>
    `;
    
    container.innerHTML = html;
}

function displayDashboardResults(results, target, days, progressInfo = null) {
    const container = document.getElementById('resultsContainer');

    if (results.length === 0) {
        container.innerHTML = '<div class="empty-state">No results found</div>';
        return;
    }

    // Show progress header if loading more
    let progressHeader = '';
    if (progressInfo && progressInfo.isPartial) {
        const percentage = Math.round((progressInfo.loaded / progressInfo.total) * 100);
        progressHeader = `
            <div class="results-progress-header loading">
                <span class="progress-count">
                    📊 Loading <strong>${progressInfo.loaded}</strong> of <strong>${progressInfo.total}</strong> results...
                </span>
                <span class="progress-bar-container">
                    <span class="progress-bar" style="width: ${percentage}%"></span>
                </span>
                <span class="progress-percentage">${percentage}%</span>
            </div>
        `;
    } else {
        progressHeader = `
            <div class="results-progress-header">
                <span class="progress-count">
                    ✅ <strong>${results.length}</strong> result${results.length !== 1 ? 's' : ''} loaded
                </span>
            </div>
        `;
    }

    let html = progressHeader + `
        <table class="results-table">
            <thead>
                <tr>
                    <th class="col-no">No.</th>
                    <th class="col-symbol">Company Symbol</th>
                    <th class="col-indicator">Indicator</th>
                    <th class="col-total">Total Signals</th>
                    <th class="col-success">Success</th>
                    <th class="col-failure">Failure</th>
                    <th class="col-open">Open</th>
                    <th class="col-rate">Success %</th>
                    <th class="col-action">Action</th>
                </tr>
            </thead>
            <tbody>
    `;

    results.forEach((result, index) => {
        const totalSignals = result.totalSignals || 0;
        const completedTrades = result.completedTrades || 0;
        const successful = result.successful || 0;
        const openTrades = result.openTrades || 0;
        const successRate = result.successRate || 0;
        const failureSignals = completedTrades - successful;
        const successClass = successRate >= 70 ? 'high' : successRate >= 50 ? 'medium' : 'low';
        
        html += `
            <tr>
                <td class="rank">${index + 1}</td>
                <td class="symbol">${result.symbol}</td>
                <td class="col-indicator"><span class="indicator">${result.indicator}</span></td>
                <td class="center">${totalSignals}</td>
                <td class="center"><span class="badge badge-success">${successful}</span></td>
                <td class="center"><span class="badge badge-failure">${failureSignals}</span></td>
                <td class="center"><span class="badge badge-open">${openTrades}</span></td>
                <td class="center"><span class="badge badge-rate success-rate-${successClass}">${successRate}%</span></td>
                <td class="center">
                    <a href="/symbol/${result.symbol}?indicator=${encodeURIComponent(result.indicator)}" class="btn-view">
                        VIEW DETAILS
                    </a>
                </td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

// ============================================
// LOAD AND DISPLAY AVAILABLE INDICATORS
// ============================================
// LOAD AND DISPLAY AVAILABLE INDICATORS
// ============================================
// LOAD DASHBOARD SIGNALS ON PAGE LOAD
// ============================================
async function loadDashboardSignals() {
    try {
        console.log('📊 [DASHBOARD] Loading current BUY signals...');
        
        // Load summary info
        const summaryResponse = await fetch(API.summary);
        const summaryData = await summaryResponse.json();
        
        // Update summary display if element exists
        const summaryDiv = document.getElementById('dashboardSummary');
        if (summaryDiv) {
            summaryDiv.innerHTML = `
                <div class="summary-card">
                    <h3>Current Signals</h3>
                    <p class="big-number">${summaryData.buy || 0}</p>
                    <p class="label">BUY Signals</p>
                    <p class="date">Date: ${summaryData.date || 'N/A'}</p>
                </div>
            `;
        }
        
        console.log(`✅ [DASHBOARD] Loaded ${summaryData.buy} BUY signals for ${summaryData.date}`);
        
    } catch (error) {
        console.error('❌ [DASHBOARD] Error loading signals:', error);
    }
}

// ============================================
// LOAD AND DISPLAY AVAILABLE INDICATORS
// ============================================
async function loadAvailableIndicators() {
    try {
        // Fetch indicators that have BUY signals with their counts
        const response = await fetch('/api/signals/by-indicator');
        const data = await response.json();
        
        const countElement = document.getElementById('indicatorCount');
        const listElement = document.getElementById('indicatorsList');
        
        if (!countElement || !listElement) return;
        
        const indicators = data.indicators || {};
        const indicatorNames = Object.keys(indicators);
        
        // Update count
        countElement.textContent = `${indicatorNames.length}`;
        
        // Group indicators by type
        const smaIndicators = indicatorNames.filter(i => i.startsWith('SMA')).sort();
        const rsiIndicators = indicatorNames.filter(i => i.startsWith('RSI')).sort();
        const bbIndicators = indicatorNames.filter(i => i.startsWith('BB')).sort();
        const macdIndicators = indicatorNames.filter(i => ['Short', 'Long', 'Standard'].includes(i)).sort();
        const stochIndicators = indicatorNames.filter(i => i.startsWith('STOCH')).sort();
        
        let html = '';
        
        // SMA badges
        if (smaIndicators.length > 0) {
            html += smaIndicators.map(ind => 
                `<span class="indicator-badge sma" title="${indicators[ind]} BUY signals">${ind}</span>`
            ).join('');
        }
        
        // RSI badges
        if (rsiIndicators.length > 0) {
            html += rsiIndicators.map(ind => 
                `<span class="indicator-badge rsi" title="${indicators[ind]} BUY signals">${ind}</span>`
            ).join('');
        }
        
        // Bollinger Bands badges (show full name like BB10_LOWER with color coding)
        if (bbIndicators.length > 0) {
            html += bbIndicators.map(ind => {
                let badgeClass = 'indicator-badge bb';
                if (ind.includes('Lower')) badgeClass += ' bb-lower';
                else if (ind.includes('Upper')) badgeClass += ' bb-upper';
                else if (ind.includes('Middle')) badgeClass += ' bb-middle';
                
                return `<span class="${badgeClass}" title="${indicators[ind]} BUY signals">${ind}</span>`;
            }).join('');
        }
        
        // MACD badges (display as MACD_Short, MACD_Long, MACD_Standard)
        if (macdIndicators.length > 0) {
            html += macdIndicators.map(ind => 
                `<span class="indicator-badge macd" title="${indicators[ind]} BUY signals">MACD_${ind}</span>`
            ).join('');
        }
        
        // Stochastic badges
        if (stochIndicators.length > 0) {
            html += stochIndicators.map(ind => 
                `<span class="indicator-badge stoch" title="${indicators[ind]} BUY signals">${ind}</span>`
            ).join('');
        }
        
        listElement.innerHTML = html || '<div class="empty-state">No indicators with BUY signals</div>';
        
    } catch (error) {
        console.error('Failed to load indicators:', error);
        const listElement = document.getElementById('indicatorsList');
        if (listElement) {
            listElement.innerHTML = '<div class="error-state">Failed to load indicators</div>';
        }
    }
}

// Toggle indicators visibility
function setupIndicatorsToggle() {
    const toggle = document.getElementById('indicatorsToggle');
    const list = document.getElementById('indicatorsList');
    
    if (!toggle || !list) return;
    
    toggle.addEventListener('click', function(e) {
        e.stopPropagation();
        const isHidden = list.classList.contains('hidden');
        
        if (isHidden) {
            // Simply toggle visibility - CSS handles positioning
            list.classList.remove('hidden');
            toggle.classList.add('expanded');
        } else {
            list.classList.add('hidden');
            toggle.classList.remove('expanded');
        }
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (!toggle.contains(e.target) && !list.contains(e.target)) {
            list.classList.add('hidden');
            toggle.classList.remove('expanded');
        }
    });
}

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('dashboardAnalyzeBtn')) {
        // Dashboard page
        const analyzeBtn = document.getElementById('dashboardAnalyzeBtn');
        const groupCheckbox = document.getElementById('groupByCompany');
        
        // Use progressive loading if available, otherwise fall back to regular
        if (typeof analyzeDashboardProgressive !== 'undefined') {
            analyzeBtn.addEventListener('click', function() {
                // Check if grouped analysis is requested
                if (groupCheckbox && groupCheckbox.checked && typeof analyzeDashboardGrouped !== 'undefined') {
                    analyzeDashboardGrouped();
                } else if (typeof analyzeDashboardFast !== 'undefined') {
                    analyzeDashboardFast();
                } else {
                    analyzeDashboardProgressive();
                }
            });
        } else {
            analyzeBtn.addEventListener('click', analyzeDashboard);
        }
        
        // Add change event listener for group by company checkbox
        if (groupCheckbox) {
            groupCheckbox.addEventListener('change', function() {
                // Re-analyze when toggle changes (if results already loaded)
                if (allResults && allResults.length > 0) {
                    if (this.checked && typeof analyzeDashboardGrouped !== 'undefined') {
                        analyzeDashboardGrouped();
                    } else if (typeof analyzeDashboardFast !== 'undefined') {
                        analyzeDashboardFast();
                    }
                }
            });
        }

        // Load summary info
        loadSummaryInfo();
        
        // Load available indicators
        loadAvailableIndicators();
        
        // Setup indicators toggle
        setupIndicatorsToggle();
        
        // Load signals immediately on page load
        loadDashboardSignals();
    } else if (document.getElementById('totalSymbols')) {
        loadDashboardData();
        
        const refreshBtn = document.getElementById('refreshBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                loadDashboardData();
            });
        }
    }
    
    if (typeof SYMBOL !== 'undefined') {
        // Don't load indicator table on page load - it's slow
        // loadSymbolIndicators();
        
        // Load chart indicators first, then update chart with defaults
        loadChartIndicators().then(() => {
            console.log('📊 [INIT] Chart indicators loaded, setting defaults...');
            
            // Set default indicators for initial chart display
            const smaSelect = document.getElementById('smaSelect');
            const rsiSelect = document.getElementById('rsiSelect');
            const bbSelect = document.getElementById('bbSelect');
            const macdSelect = document.getElementById('macdSelect');
            const stochSelect = document.getElementById('stochSelect');
            
            // Set default values if available
            if (smaSelect && smaSelect.options.length > 1) {
                smaSelect.selectedIndex = 3; // SMA20 (index 0 is "Select SMA", so index 3 is SMA20)
            }
            if (rsiSelect && rsiSelect.options.length > 1) {
                rsiSelect.selectedIndex = 2; // RSI14
            }
            if (bbSelect && bbSelect.options.length > 1) {
                bbSelect.selectedIndex = 2; // BB20
            }
            if (macdSelect && macdSelect.options.length > 1) {
                macdSelect.selectedIndex = 3; // Standard
            }
            if (stochSelect && stochSelect.options.length > 1) {
                stochSelect.selectedIndex = 3; // STOCH14
            }
            
            // After indicators are loaded and defaults set, update chart
            console.log('📊 [INIT] Updating chart with default indicators...');
            updateChart();
            
            // NOW check URL parameters and auto-run analysis (AFTER indicators are loaded)
            const params = new URLSearchParams(window.location.search);
            const urlIndicator = params.get('indicator');
            
            console.log(`📊 [AUTO] URL search params: ${window.location.search}`);
            console.log(`📊 [AUTO] URL indicator parameter: ${urlIndicator}`);
            
            if (urlIndicator) {
                // URL has indicator parameter - use it
                console.log(`📊 [AUTO] Found URL indicator: ${urlIndicator}`);
                const analysisSelect = document.getElementById('analysisIndicator');
                if (analysisSelect) {
                    // Check if this indicator exists in the dropdown
                    let found = false;
                    for (let i = 0; i < analysisSelect.options.length; i++) {
                        if (analysisSelect.options[i].value === urlIndicator) {
                            found = true;
                            break;
                        }
                    }
                    
                    if (found) {
                        analysisSelect.value = urlIndicator;
                        console.log(`📊 [AUTO] Set analysis dropdown to: ${urlIndicator}`);
                        
                        // Auto-run analysis after a short delay
                        setTimeout(() => {
                            console.log(`📊 [AUTO] Running auto-analysis for ${urlIndicator}`);
                            analyzeSymbol();
                        }, 500);
                    } else {
                        console.error(`❌ [AUTO] Indicator "${urlIndicator}" not found in dropdown`);
                        console.log(`📊 [AUTO] Available indicators:`, Array.from(analysisSelect.options).map(o => o.value));
                        // Fall back to first indicator
                        if (analysisSelect.options.length > 1) {
                            analysisSelect.selectedIndex = 1;
                            console.log(`📊 [AUTO] Falling back to: ${analysisSelect.value}`);
                            setTimeout(() => {
                                analyzeSymbol();
                            }, 500);
                        }
                    }
                }
            } else {
                // No URL indicator - auto-select first available indicator
                console.log('📊 [AUTO] No URL indicator, auto-selecting first available...');
                const analysisSelect = document.getElementById('analysisIndicator');
                if (analysisSelect && analysisSelect.options.length > 1) {
                    // Select the first real indicator (skip the placeholder at index 0)
                    analysisSelect.selectedIndex = 1;
                    console.log(`📊 [AUTO] Auto-selected: ${analysisSelect.value}`);
                    
                    // Auto-run analysis with default values
                    const targetInput = document.getElementById('analysisTarget');
                    const daysInput = document.getElementById('analysisDays');
                    
                    if (targetInput && daysInput && targetInput.value && daysInput.value) {
                        console.log(`📊 [AUTO] Running auto-analysis with target=${targetInput.value}%, days=${daysInput.value}`);
                        setTimeout(() => {
                            analyzeSymbol();
                        }, 500);
                    }
                } else {
                    console.log('⚠️ [AUTO] Analysis dropdown not populated or empty');
                }
            }
        }).catch(error => {
            console.error('❌ [INIT] Failed to load chart indicators:', error);
        });
        
        const updateChartBtn = document.getElementById('updateChartBtn');
        if (updateChartBtn) {
            updateChartBtn.addEventListener('click', updateChart);
        }
        
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', analyzeSymbol);
        }
        
        // Load indicators on demand
        const loadIndicatorsBtn = document.getElementById('loadIndicatorsBtn');
        if (loadIndicatorsBtn) {
            loadIndicatorsBtn.addEventListener('click', function() {
                loadIndicatorsBtn.style.display = 'none';
                document.getElementById('indicatorsTableWrapper').style.display = 'block';
                loadSymbolIndicators();
            });
        }
    }
});

window.addEventListener('resize', function() {
    if (priceChart) {
        priceChart.applyOptions({
            width: document.getElementById('priceChart').clientWidth
        });
    }
    if (bbChart) {
        bbChart.applyOptions({
            width: document.getElementById('bbChart').clientWidth
        });
    }
    if (rsiChart) {
        rsiChart.applyOptions({
            width: document.getElementById('rsiChart').clientWidth
        });
    }
    if (macdChart) {
        macdChart.applyOptions({
            width: document.getElementById('macdChart').clientWidth
        });
    }
    if (stochChart) {
        stochChart.applyOptions({
            width: document.getElementById('stochChart').clientWidth
        });
    }
});

const style = document.createElement('style');
style.textContent = `
    body.no-animation * {
        animation: none !important;
        transition: none !important;
    }
`;
document.head.appendChild(style);
