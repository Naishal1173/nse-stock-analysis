// FAST SCANNER - Loads all results at once with caching
let allScanResults = [];
let currentScanPage = 1;
let SCAN_ITEMS_PER_PAGE = 100; // Default 100 per page
let currentScanParams = {};
let isGroupedView = false;
let currentSearchTerm = '';
let currentSortBy = 'success'; // Default to highest success rate

// Cache management
let cachedScanData = null;
let cacheTimestamp = null;
const CACHE_TTL = 300000; // 5 minutes

// Log cache status on load
console.log('🚀 [SCANNER] Fast scanner initialized');
console.log(`⏱️  [SCANNER] Cache TTL: ${CACHE_TTL / 1000}s`);

// Declare functions that will be used globally
function changeScanPage(page) {
    console.log(`📄 [PAGINATION] Changing to page ${page}`);
    const totalPages = Math.ceil(allScanResults.length / SCAN_ITEMS_PER_PAGE);
    if (page < 1 || page > totalPages) {
        console.log(`⚠️  [PAGINATION] Invalid page ${page}, valid range: 1-${totalPages}`);
        return;
    }
    
    currentScanPage = page;
    displayScanResults(allScanResults, false);
    
    // Scroll to top of results
    document.getElementById('scannerResults').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Make function globally accessible immediately
window.changeScanPage = changeScanPage;

// Verify it's accessible
console.log('✅ [SCANNER] changeScanPage function is globally accessible:', typeof window.changeScanPage);

async function performScanFast() {
    const target = document.getElementById('scannerTarget').value;
    const stopLoss = document.getElementById('scannerStopLoss').value;
    const holdingDays = document.getElementById('holdingDays').value || '30';
    const fromDate = document.getElementById('scannerFromDate').value || '';
    const toDate = document.getElementById('scannerToDate').value || '';
    const indicator = document.getElementById('indicatorFilter').value || 'ALL';
    
    if (!target || !stopLoss || !holdingDays) {
        showNotification('Please fill all required fields', 'warning');
        return;
    }
    
    // Store current parameters
    currentScanParams = { target, stopLoss, holdingDays, fromDate, toDate, indicator };
    
    // Create cache key that matches server-side cache key format
    const cacheKey = `${target}_${stopLoss}_${holdingDays}_${fromDate}_${toDate}_${indicator}`;
    const now = Date.now();
    
    // Check if we have cached data with same parameters
    if (cachedScanData && cachedScanData.cacheKey === cacheKey && 
        cacheTimestamp && (now - cacheTimestamp) < CACHE_TTL) {
        console.log('📦 [SCANNER] Using client-side cached data');
        allScanResults = cachedScanData.results;
        currentScanPage = 1; // Reset to page 1 for cached results
        displayScanResults(allScanResults, true);
        showNotification(`Found ${allScanResults.length} companies (from client cache)`, 'success');
        return;
    }
    
    // If cache miss or expired, clear old cache
    if (cachedScanData && cachedScanData.cacheKey !== cacheKey) {
        console.log('🔄 [SCANNER] Parameters changed, clearing old cache');
        cachedScanData = null;
        cacheTimestamp = null;
    }
    
    // Show loading
    document.getElementById('scannerLoading').style.display = 'block';
    document.getElementById('scannerResults').style.display = 'none';
    
    // Update loading message with indicator info
    const loadingProgress = document.getElementById('scanProgress');
    if (loadingProgress) {
        if (indicator === 'ALL') {
            loadingProgress.textContent = 'Analyzing latest BUY signals across all indicators...';
        } else {
            loadingProgress.textContent = `Analyzing latest ${indicator} BUY signals...`;
        }
    }
    
    try {
        // Always use latest_only=true for better performance and relevance
        const url = `/api/day-trading-scan?target=${target}&stop_loss=${stopLoss}&holding_days=${holdingDays}&from_date=${fromDate}&to_date=${toDate}&indicator=${indicator}&latest_only=true`;
        console.log('🔍 [SCANNER] Fetching:', url);
        console.log('🔑 [SCANNER] Cache key:', cacheKey);
        
        const startTime = performance.now();
        const response = await fetch(url);
        const data = await response.json();
        const endTime = performance.now();
        const loadTime = ((endTime - startTime) / 1000).toFixed(2);
        
        if (data.error) {
            showNotification('Scan failed: ' + data.error, 'error');
            return;
        }
        
        allScanResults = data.results || [];
        
        // Reset to page 1 for new results
        currentScanPage = 1;
        
        // Cache the results with the correct key
        cachedScanData = {
            cacheKey: cacheKey,
            results: allScanResults,
            timestamp: now
        };
        cacheTimestamp = now;
        
        console.log(`✅ [SCANNER] Loaded ${allScanResults.length} results in ${loadTime}s`);
        console.log(`💾 [SCANNER] Cached with key: ${cacheKey}`);
        
        displayScanResults(allScanResults, data.cached);
        
        // Show appropriate notification
        if (data.cached) {
            showNotification(`Found ${allScanResults.length} companies in ${loadTime}s (server cache)`, 'success');
        } else {
            showNotification(`Found ${allScanResults.length} companies in ${loadTime}s`, 'success');
        }
        
    } catch (error) {
        console.error('❌ [SCANNER] Error:', error);
        showNotification('Scan failed: ' + error.message, 'error');
    } finally {
        document.getElementById('scannerLoading').style.display = 'none';
    }
}

function displayScanResults(results, isCached = false) {
    console.log(`📊 [DISPLAY] Displaying ${results.length} results, isCached=${isCached}, grouped=${isGroupedView}, search="${currentSearchTerm}"`);
    
    // DON'T reset to page 1 here - it breaks pagination!
    // Only reset when new scan results come in (handled in performScanFast)
    
    document.getElementById('scannerResults').style.display = 'block';
    
    // Apply search filter first
    let filteredResults = applySearchFilter(results);
    
    // Apply grouping if enabled
    if (isGroupedView) {
        filteredResults = groupResultsByCompany(filteredResults);
    }
    
    document.getElementById('totalCompanies').textContent = isGroupedView ? filteredResults.length : results.length;
    
    // Show/hide cache indicator with more detail
    const cacheIndicator = document.getElementById('cacheIndicator');
    if (isCached) {
        cacheIndicator.style.display = 'inline';
        cacheIndicator.textContent = '💾 Server Cached';
        cacheIndicator.title = 'Results from server cache (24h)';
    } else if (cachedScanData && cachedScanData.results === results) {
        cacheIndicator.style.display = 'inline';
        cacheIndicator.textContent = '⚡ Client Cached';
        cacheIndicator.title = 'Results from browser cache (5min)';
    } else {
        cacheIndicator.style.display = 'none';
    }
    
    const totalProfitSignals = filteredResults.reduce((sum, r) => sum + r.profit_signals, 0);
    document.getElementById('totalProfitSignals').textContent = totalProfitSignals;
    
    // Apply current sort (default: success rate)
    filteredResults = applyScanSort(filteredResults, currentSortBy);
    
    console.log(`📊 [DISPLAY] After sort: ${filteredResults.length} results`);
    
    // Pagination
    const totalPages = Math.ceil(filteredResults.length / SCAN_ITEMS_PER_PAGE);
    const startIndex = (currentScanPage - 1) * SCAN_ITEMS_PER_PAGE;
    const endIndex = startIndex + SCAN_ITEMS_PER_PAGE;
    const paginatedResults = filteredResults.slice(startIndex, endIndex);
    
    console.log(`📊 [DISPLAY] Page ${currentScanPage}/${totalPages}, showing ${paginatedResults.length} items (${startIndex}-${endIndex})`);
    
    const tbody = document.getElementById('resultsTableBody');
    tbody.innerHTML = '';
    
    if (filteredResults.length === 0) {
        const message = currentSearchTerm 
            ? `No results found for "${currentSearchTerm}"`
            : 'No companies found. Try adjusting the parameters or date range.';
        tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 2rem;">${message}</td></tr>`;
        updatePaginationControls(0, 0);
        return;
    }
    
    console.log(`📊 [DISPLAY] Rendering ${paginatedResults.length} rows...`);
    
    if (isGroupedView) {
        renderGroupedResults(paginatedResults, tbody, startIndex);
    } else {
        renderUngroupedResults(paginatedResults, tbody, startIndex);
    }
    
    console.log(`✅ [DISPLAY] Rendered ${paginatedResults.length} rows successfully`);
    
    updatePaginationControls(filteredResults.length, totalPages);
    
    // Scroll to results section
    setTimeout(() => {
        document.getElementById('scannerResults').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function renderUngroupedResults(results, tbody, startIndex) {
    results.forEach((result, index) => {
        const globalIndex = startIndex + index + 1;
        const row = document.createElement('tr');
        
        // Success rate based on TOTAL signals (including open trades)
        const profitSignals = result.profit_signals || 0;
        const lossSignals = result.loss_signals || 0;
        const completedTrades = profitSignals + lossSignals;
        const openTrades = result.open_trades || 0;
        const totalSignals = result.total_signals || 0;
        
        // Calculate success rate from total signals (including open trades)
        const successRate = totalSignals > 0 ? (profitSignals / totalSignals * 100) : 0;
        
        // Net P/L is the average from the backend
        const netProfit = result.net_profit_loss || 0;
        
        const successClass = successRate >= 70 ? 'success-high' : successRate >= 50 ? 'success-medium' : 'success-low';
        const profitClass = netProfit >= 0 ? 'profit' : 'loss';
        
        const indicator = result.indicator || (result.indicators && result.indicators[0]) || '-';
        const indicatorParam = indicator !== '-' ? indicator : 'RSI7';
        
        row.innerHTML = `
            <td>${globalIndex}</td>
            <td><strong>${result.symbol}</strong></td>
            <td class="center">${indicator}</td>
            <td class="center">${totalSignals}</td>
            <td class="center profit">${profitSignals}</td>
            <td class="center loss">${lossSignals}</td>
            <td class="center">${openTrades}</td>
            <td class="center"><span class="badge ${successClass}">${successRate.toFixed(1)}%</span></td>
            <td class="center ${profitClass}"><strong>${netProfit >= 0 ? '+' : ''}${netProfit.toFixed(2)}%</strong></td>
            <td class="center">
                <a href="/scanner-detail/${encodeURIComponent(result.symbol)}?indicator=${encodeURIComponent(indicatorParam)}&target=${currentScanParams.target}&stop_loss=${currentScanParams.stopLoss}&days=${currentScanParams.holdingDays}&from_date=${currentScanParams.fromDate || ''}&to_date=${currentScanParams.toDate || ''}" 
                   class="btn-view" target="_blank">VIEW</a>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

function renderGroupedResults(groups, tbody, startIndex) {
    groups.forEach((group, index) => {
        const globalIndex = startIndex + index + 1;
        const indicatorCount = group.indicators.length;
        
        // CORRECTED: Success rate = (profit_signals / total_signals) * 100
        const totalSignals = group.total_signals || 0;
        const profitSignals = group.profit_signals || 0;
        const lossSignals = group.loss_signals || 0;
        const openTrades = group.open_trades || 0;
        
        // Success rate based on total signals (including open trades)
        const successRate = totalSignals > 0 ? (profitSignals / totalSignals * 100) : 0;
        
        const successClass = successRate >= 70 ? 'success-high' : successRate >= 50 ? 'success-medium' : 'success-low';
        const profitClass = group.avg_net_profit >= 0 ? 'profit' : 'loss';
        
        // Main row with company name and aggregated stats
        const mainRow = document.createElement('tr');
        mainRow.className = 'company-group-start';
        mainRow.innerHTML = `
            <td>${globalIndex}</td>
            <td><strong>${group.symbol}</strong> <span class="indicator-count-badge">${indicatorCount} indicators</span></td>
            <td class="center">
                ${group.indicators.map(ind => `<span class="indicator-badge">${ind.indicator}</span>`).join(' ')}
            </td>
            <td class="center">${totalSignals}</td>
            <td class="center profit">${profitSignals}</td>
            <td class="center loss">${lossSignals}</td>
            <td class="center">${openTrades}</td>
            <td class="center"><span class="badge ${successClass}">${successRate.toFixed(1)}%</span></td>
            <td class="center ${profitClass}"><strong>${group.avg_net_profit >= 0 ? '+' : ''}${group.avg_net_profit.toFixed(2)}%</strong></td>
            <td class="center">
                <a href="/scanner-detail/${encodeURIComponent(group.symbol)}?indicator=${encodeURIComponent(group.indicators[0].indicator)}&target=${currentScanParams.target}&stop_loss=${currentScanParams.stopLoss}&days=${currentScanParams.holdingDays}&from_date=${currentScanParams.fromDate || ''}&to_date=${currentScanParams.toDate || ''}" 
                   class="btn-view" target="_blank">VIEW</a>
            </td>
        `;
        tbody.appendChild(mainRow);
    });
}

function applyScanSort(results, sortBy) {
    const sorted = [...results];
    sorted.sort((a, b) => {
        switch(sortBy) {
            case 'signals':
                return b.profit_signals - a.profit_signals;
            case 'success':
                return (b.success_rate || 0) - (a.success_rate || 0);
            case 'netProfit':
                return (b.net_profit_loss || 0) - (a.net_profit_loss || 0);
            case 'symbol':
                return a.symbol.localeCompare(b.symbol);
            default:
                return 0;
        }
    });
    return sorted;
}

function updatePaginationControls(totalResults, totalPages) {
    const paginationInfo = document.getElementById('paginationInfo');
    const paginationButtons = document.getElementById('paginationButtons');
    const paginationBottomButtons = document.getElementById('paginationBottomButtons');
    
    if (totalResults === 0) {
        paginationInfo.textContent = '';
        paginationButtons.innerHTML = '';
        paginationBottomButtons.innerHTML = '';
        return;
    }
    
    const startIndex = (currentScanPage - 1) * SCAN_ITEMS_PER_PAGE + 1;
    const endIndex = Math.min(currentScanPage * SCAN_ITEMS_PER_PAGE, totalResults);
    
    paginationInfo.textContent = `Showing ${startIndex}-${endIndex} of ${totalResults}`;
    
    const buttonsHTML = `
        <button onclick="changeScanPage(1)" ${currentScanPage === 1 ? 'disabled' : ''} 
                style="padding: 6px 12px; margin: 0 2px; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer;">« First</button>
        <button onclick="changeScanPage(${currentScanPage - 1})" ${currentScanPage === 1 ? 'disabled' : ''}
                style="padding: 6px 12px; margin: 0 2px; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer;">‹ Prev</button>
        <span style="padding: 6px 12px; margin: 0 8px;">Page ${currentScanPage} of ${totalPages}</span>
        <button onclick="changeScanPage(${currentScanPage + 1})" ${currentScanPage === totalPages ? 'disabled' : ''}
                style="padding: 6px 12px; margin: 0 2px; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer;">Next ›</button>
        <button onclick="changeScanPage(${totalPages})" ${currentScanPage === totalPages ? 'disabled' : ''}
                style="padding: 6px 12px; margin: 0 2px; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer;">Last »</button>
    `;
    
    paginationButtons.innerHTML = buttonsHTML;
    paginationBottomButtons.innerHTML = buttonsHTML;
}

function applySortAndFilter() {
    if (allScanResults.length === 0) return;
    
    currentScanPage = 1; // Reset to first page
    displayScanResults(allScanResults, false);
}

// Search functionality
function setupScannerSearch() {
    const searchInput = document.getElementById('scannerSearch');
    const clearBtn = document.getElementById('clearScannerSearch');
    
    if (!searchInput) return;
    
    searchInput.addEventListener('input', function() {
        currentSearchTerm = this.value.toLowerCase().trim();
        
        if (clearBtn) {
            clearBtn.style.display = currentSearchTerm ? 'block' : 'none';
        }
        
        currentScanPage = 1;
        displayScanResults(allScanResults, false);
    });
    
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            searchInput.value = '';
            currentSearchTerm = '';
            clearBtn.style.display = 'none';
            currentScanPage = 1;
            displayScanResults(allScanResults, false);
        });
    }
}

// Group by company functionality
function groupResultsByCompany(results) {
    const grouped = {};
    
    results.forEach(result => {
        if (!grouped[result.symbol]) {
            grouped[result.symbol] = {
                symbol: result.symbol,
                indicators: [],
                total_signals: 0,
                profit_signals: 0,
                loss_signals: 0,
                open_trades: 0,
                max_success_rate: 0,
                avg_net_profit: 0
            };
        }
        
        const group = grouped[result.symbol];
        group.indicators.push({
            indicator: result.indicator,
            total_signals: result.total_signals,
            profit_signals: result.profit_signals,
            loss_signals: result.loss_signals,
            open_trades: result.open_trades,
            success_rate: result.success_rate,
            net_profit_loss: result.net_profit_loss
        });
        
        group.total_signals += result.total_signals;
        group.profit_signals += result.profit_signals;
        group.loss_signals += result.loss_signals;
        group.open_trades += result.open_trades;
        group.max_success_rate = Math.max(group.max_success_rate, result.success_rate || 0);
    });
    
    // Calculate average net profit for each group
    Object.values(grouped).forEach(group => {
        const totalProfit = group.indicators.reduce((sum, ind) => sum + (ind.net_profit_loss || 0), 0);
        group.avg_net_profit = totalProfit / group.indicators.length;
    });
    
    return Object.values(grouped);
}

function applySearchFilter(results) {
    if (!currentSearchTerm) return results;
    
    return results.filter(result => {
        const symbol = result.symbol.toLowerCase();
        
        // Handle both single indicator and array of indicators (for grouped results)
        let indicatorMatch = false;
        if (result.indicator) {
            // Ungrouped result - single indicator
            indicatorMatch = result.indicator.toLowerCase().includes(currentSearchTerm);
        } else if (result.indicators && Array.isArray(result.indicators)) {
            // Grouped result - array of indicators
            indicatorMatch = result.indicators.some(ind => {
                const indName = typeof ind === 'string' ? ind : ind.indicator;
                return indName && indName.toLowerCase().includes(currentSearchTerm);
            });
        }
        
        return symbol.includes(currentSearchTerm) || indicatorMatch;
    });
}

function resetForm() {
    document.getElementById('scannerTarget').value = '5';
    document.getElementById('scannerStopLoss').value = '3';
    document.getElementById('holdingDays').value = '30';
    document.getElementById('indicatorFilter').value = 'ALL';
    
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    
    document.getElementById('scannerToDate').valueAsDate = today;
    document.getElementById('scannerFromDate').valueAsDate = oneYearAgo;
    
    document.getElementById('scannerResults').style.display = 'none';
    allScanResults = [];
    cachedScanData = null;
    cacheTimestamp = null;
    
    showNotification('Form reset to defaults', 'info');
}

async function clearCache() {
    try {
        // Clear client-side cache first
        const hadClientCache = cachedScanData !== null;
        cachedScanData = null;
        cacheTimestamp = null;
        
        console.log('🗑️ [SCANNER] Clearing server cache...');
        
        const response = await fetch('/api/clear-scanner-cache', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.success) {
            const message = hadClientCache 
                ? 'Both client and server cache cleared successfully' 
                : data.message;
            showNotification(message, 'success');
            console.log('✅ [SCANNER] Cache cleared');
        } else {
            showNotification('Failed to clear server cache', 'error');
        }
    } catch (error) {
        console.error('❌ [CACHE] Error:', error);
        showNotification('Failed to clear cache: ' + error.message, 'error');
    }
}

function showNotification(message, type = 'info') {
    const color = type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#3b82f6';
    console.log(`[${type.toUpperCase()}] ${message}`);
    const notification = document.createElement('div');
    notification.style.cssText = `position: fixed; top: 20px; right: 20px; background: ${color}; color: white; padding: 1rem 1.5rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); z-index: 10000; max-width: 400px; font-size: 14px;`;
    notification.textContent = message;
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 3000);
}

// Debug function to check cache status
function getCacheStatus() {
    if (!cachedScanData) {
        console.log('📊 [CACHE STATUS] No cache data');
        return;
    }
    
    const now = Date.now();
    const age = Math.round((now - cacheTimestamp) / 1000);
    const ttl = Math.round(CACHE_TTL / 1000);
    const remaining = Math.max(0, ttl - age);
    
    console.log('📊 [CACHE STATUS]');
    console.log(`  Cache Key: ${cachedScanData.cacheKey}`);
    console.log(`  Results: ${cachedScanData.results.length} companies`);
    console.log(`  Age: ${age}s / ${ttl}s`);
    console.log(`  Remaining: ${remaining}s`);
    console.log(`  Valid: ${remaining > 0 ? '✅' : '❌'}`);
}

// Make it available globally for debugging
window.getCacheStatus = getCacheStatus;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    const scanBtn = document.getElementById('scanBtn');
    const resetBtn = document.getElementById('resetBtn');
    const clearCacheBtn = document.getElementById('clearCacheBtn');
    const indicatorFilter = document.getElementById('indicatorFilter');
    
    // Set default dates (1 year ago to today)
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    
    document.getElementById('scannerToDate').valueAsDate = today;
    document.getElementById('scannerFromDate').valueAsDate = oneYearAgo;
    
    if (scanBtn) {
        scanBtn.addEventListener('click', performScanFast);
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', resetForm);
    }
    
    if (clearCacheBtn) {
        clearCacheBtn.addEventListener('click', clearCache);
    }
    
    if (indicatorFilter) {
        indicatorFilter.addEventListener('change', function() {
            console.log('🔄 [SCANNER] Indicator changed to:', this.value);
            performScanFast();
        });
    }
    
    // Setup search functionality
    setupScannerSearch();
    
    // Setup group by company toggle
    const groupByCheckbox = document.getElementById('groupByCompany');
    if (groupByCheckbox) {
        groupByCheckbox.addEventListener('change', function() {
            isGroupedView = this.checked;
            currentScanPage = 1;
            if (allScanResults.length > 0) {
                displayScanResults(allScanResults, false);
            }
        });
    }
});
