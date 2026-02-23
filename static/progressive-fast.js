// FAST VERSION - Loads all results at once
let allResults = [];
let currentDisplayedResults = []; // Track what's currently being displayed
let dashboardCurrentPage = 1;
const ITEMS_PER_PAGE = 50;
let selectedIndicators = [];

async function analyzeDashboardFast() {
    const target = document.getElementById('dashboardTarget').value;
    const days = document.getElementById('dashboardDays').value;

    if (!target || !days) {
        showNotification('Fill target and days', 'warning');
        return;
    }

    try {
        allResults = [];
        dashboardCurrentPage = 1;

        document.getElementById('loadingState').classList.remove('hidden');
        document.getElementById('resultsSection').classList.add('hidden');

        console.log('📊 [DASHBOARD] Starting PROGRESSIVE analysis...');
        const startTime = performance.now();

        // STEP 1: Get first 50 results (backend analyzes ALL, returns first 50)
        console.log('📊 [DASHBOARD] Loading first 50 results...');
        const firstUrl = `/api/analyze-progressive?target=${target}&days=${days}&batch_size=50&offset=0`;
        const firstResponse = await fetch(firstUrl);
        const firstData = await firstResponse.json();

        if (firstData.error) {
            showError('Analysis failed: ' + firstData.error);
            document.getElementById('loadingState').classList.add('hidden');
            return;
        }

        allResults = firstData.results || [];
        const totalSignals = firstData.total_signals;

        console.log(`✅ [DASHBOARD] First ${allResults.length} results loaded in ${firstData.processing_time_seconds}s`);
        
        // Show first 50 immediately
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');
        
        let filteredResults = allResults;
        if (selectedIndicators.length > 0) {
            filteredResults = allResults.filter(r => {
                let resultIndicator = r.indicator;
                if (resultIndicator === 'Short' || resultIndicator === 'Long' || resultIndicator === 'Standard') {
                    resultIndicator = `MACD_${resultIndicator}`;
                }
                return selectedIndicators.includes(resultIndicator);
            });
        }
        filteredResults = applySmartSorting(filteredResults);
        
        displayResultsWithProgress(filteredResults, target, days, {
            isPartial: true,
            loaded: allResults.length,
            total: totalSignals
        });

        // STEP 2: Load ALL remaining results in ONE request (from cache, instant)
        if (firstData.has_more) {
            console.log(`📊 [DASHBOARD] Loading remaining ${totalSignals - 50} results...`);
            
            // Request ALL remaining results at once (batch_size = total - 50)
            const remainingSize = totalSignals - 50;
            const remainingUrl = `/api/analyze-progressive?target=${target}&days=${days}&batch_size=${remainingSize}&offset=50`;
            const remainingResponse = await fetch(remainingUrl);
            const remainingData = await remainingResponse.json();

            if (!remainingData.error && remainingData.results) {
                allResults = allResults.concat(remainingData.results);
                console.log(`✅ [DASHBOARD] All ${allResults.length} results loaded!`);
                
                // Apply filters and sorting to all results
                filteredResults = allResults;
                if (selectedIndicators.length > 0) {
                    filteredResults = allResults.filter(r => {
                        let resultIndicator = r.indicator;
                        if (resultIndicator === 'Short' || resultIndicator === 'Long' || resultIndicator === 'Standard') {
                            resultIndicator = `MACD_${resultIndicator}`;
                        }
                        return selectedIndicators.includes(resultIndicator);
                    });
                }
                filteredResults = applySmartSorting(filteredResults);
                allResults = filteredResults;
                
                // Final display
                displayResults(allResults, target, days);
            }
        }

        const endTime = performance.now();
        const totalTime = ((endTime - startTime) / 1000).toFixed(2);
        console.log(`✅ [DASHBOARD] Complete in ${totalTime}s`);

        // Calculate average success rate
        const avgSuccessRate = allResults.length > 0 
            ? (allResults.reduce((sum, r) => sum + r.successRate, 0) / allResults.length).toFixed(2)
            : 0;

        showNotification(`✅ Analyzed ${allResults.length} signals - Avg success: ${avgSuccessRate}%`, 'success');

    } catch (error) {
        showError('Analysis failed: ' + error.message);
        console.error('ERROR:', error);
        document.getElementById('loadingState').classList.add('hidden');
    }
}

// Filter companies by minimum number of signals
function applyMinSignalsFilter(results, minSignals) {
    if (minSignals <= 1) return results;
    
    // Count signals per company
    const symbolCounts = {};
    results.forEach(result => {
        symbolCounts[result.symbol] = (symbolCounts[result.symbol] || 0) + 1;
    });
    
    // Filter companies with at least minSignals
    return results.filter(result => symbolCounts[result.symbol] >= minSignals);
}

function applySmartSorting(results) {
    // Smart sorting logic from progressive.js
    const symbolCounts = {};
    results.forEach(result => {
        symbolCounts[result.symbol] = (symbolCounts[result.symbol] || 0) + 1;
    });

    const symbolGroups = {};
    const singleResults = [];
    
    results.forEach(result => {
        if (symbolCounts[result.symbol] > 1) {
            if (!symbolGroups[result.symbol]) {
                symbolGroups[result.symbol] = [];
            }
            symbolGroups[result.symbol].push(result);
        } else {
            singleResults.push(result);
        }
    });

    Object.keys(symbolGroups).forEach(symbol => {
        symbolGroups[symbol].sort((a, b) => {
            const rateA = a.successRate || 0;
            const rateB = b.successRate || 0;
            return rateB - rateA;
        });
    });

    const groupsWithMaxRate = Object.keys(symbolGroups).map(symbol => ({
        symbol: symbol,
        results: symbolGroups[symbol],
        maxRate: Math.max(...symbolGroups[symbol].map(r => r.successRate || 0))
    }));

    groupsWithMaxRate.sort((a, b) => {
        if (b.maxRate !== a.maxRate) return b.maxRate - a.maxRate;
        return a.symbol.localeCompare(b.symbol);
    });

    singleResults.sort((a, b) => {
        const rateA = a.successRate || 0;
        const rateB = b.successRate || 0;
        if (rateB !== rateA) return rateB - rateA;
        return a.symbol.localeCompare(b.symbol);
    });

    const sorted = [];
    let groupIndex = 0;
    let singleIndex = 0;

    while (groupIndex < groupsWithMaxRate.length || singleIndex < singleResults.length) {
        const groupMaxRate = groupIndex < groupsWithMaxRate.length ? groupsWithMaxRate[groupIndex].maxRate : -1;
        const singleRate = singleIndex < singleResults.length ? (singleResults[singleIndex].successRate || 0) : -1;

        if (groupMaxRate >= singleRate && groupIndex < groupsWithMaxRate.length) {
            sorted.push(...groupsWithMaxRate[groupIndex].results);
            groupIndex++;
        } else if (singleIndex < singleResults.length) {
            sorted.push(singleResults[singleIndex]);
            singleIndex++;
        } else {
            break;
        }
    }

    return sorted;
}

function displayResults(results, target, days) {
    const container = document.getElementById('resultsContainer');
    
    // Store current results for pagination
    currentDisplayedResults = results;
    
    const totalPages = Math.ceil(results.length / ITEMS_PER_PAGE);
    const startIndex = (dashboardCurrentPage - 1) * ITEMS_PER_PAGE;
    const endIndex = startIndex + ITEMS_PER_PAGE;
    const paginatedResults = results.slice(startIndex, endIndex);

    let html = '';

    const filterInfo = selectedIndicators.length > 0 ? ` (${selectedIndicators.length} indicators)` : '';
    // html += `<div class="results-progress-header"><span class="progress-count">✅ ${results.length} signals analyzed${filterInfo}</span><span class="page-info">Page ${dashboardCurrentPage} of ${totalPages}</span></div>`;

    html += `<table class="results-table"><thead><tr>
        <th class="col-no">No.</th>
        <th class="col-symbol">Company Symbol</th>
        <th class="col-indicator">Indicator</th>
        <th class="col-total">Total Signals</th>
        <th class="col-success">Success</th>
        <th class="col-failure">Failure</th>
        <th class="col-open">Open</th>
        <th class="col-rate">Success %</th>
        <th class="col-action">Action</th>
    </tr></thead><tbody>`;

    if (paginatedResults.length === 0) {
        html += '<tr><td colspan="9" class="center">No results</td></tr>';
    } else {
        // Count total signals per company across ALL results (not just current page)
        const totalSymbolCounts = {};
        results.forEach(result => {
            totalSymbolCounts[result.symbol] = (totalSymbolCounts[result.symbol] || 0) + 1;
        });
        
        // Group by symbol on current page
        const symbolGroups = {};
        paginatedResults.forEach((result, index) => {
            if (!symbolGroups[result.symbol]) {
                symbolGroups[result.symbol] = [];
            }
            symbolGroups[result.symbol].push({ result, index });
        });

        paginatedResults.forEach((result, index) => {
            const globalIndex = startIndex + index + 1;
            const totalSignals = result.totalSignals || 0;
            const successful = result.successful || 0;
            const failed = (result.completedTrades || 0) - successful;
            const openTrades = result.openTrades || 0;
            const successRate = result.successRate || 0;
            const successClass = successRate >= 70 ? 'high' : successRate >= 50 ? 'medium' : 'low';
            const hasNoData = totalSignals === 0;
            
            const symbolGroup = symbolGroups[result.symbol];
            const isMultiIndicator = symbolGroup.length > 1;
            const isFirstInGroup = isMultiIndicator && symbolGroup[0].index === index;
            const totalIndicatorCount = totalSymbolCounts[result.symbol]; // Total across all pages
            
            let rowClasses = [];
            if (hasNoData) rowClasses.push('no-data-row');
            if (isMultiIndicator) {
                rowClasses.push('company-group-member');
                if (isFirstInGroup) rowClasses.push('company-group-start');
            }
            const rowClass = rowClasses.join(' ');
            
            let indicatorDisplay = result.indicator;
            let indicatorParam = result.indicator;
            if (result.indicator === 'Short' || result.indicator === 'Long' || result.indicator === 'Standard') {
                indicatorDisplay = `MACD_${result.indicator}`;
            }
            
            const viewDetailsUrl = `/symbol/${encodeURIComponent(result.symbol)}?indicator=${encodeURIComponent(indicatorParam)}`;

            // Show total count badge if company has multiple signals
            const symbolDisplay = isFirstInGroup && totalIndicatorCount > 1
                ? `${result.symbol}<span class="indicator-count-badge">${totalIndicatorCount} signals</span>`
                : result.symbol;

            html += `<tr class="${rowClass}">
                <td class="rank">${globalIndex}</td>
                <td class="symbol">${symbolDisplay}</td>
                <td class="col-indicator"><div class="indicator-single">${indicatorDisplay}</div></td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">No Data</span>' : totalSignals}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-success">${successful}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-failure">${failed}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-open">${openTrades}</span>`}</td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">N/A</span>' : `<span class="badge badge-rate success-rate-${successClass}">${successRate}%</span>`}</td>
                <td class="center"><a href="${viewDetailsUrl}" class="btn-view" target="_blank" rel="noopener noreferrer">VIEW DETAILS</a></td>
            </tr>`;
        });
    }

    html += '</tbody></table>';

    if (results.length > ITEMS_PER_PAGE) {
        html += `<div class="pagination">
            <button class="pagination-btn" onclick="changePage(1)" ${dashboardCurrentPage === 1 ? 'disabled' : ''}>« First</button>
            <button class="pagination-btn" onclick="changePage(${dashboardCurrentPage - 1})" ${dashboardCurrentPage === 1 ? 'disabled' : ''}>‹ Prev</button>
            <span class="pagination-info">Page ${dashboardCurrentPage} of ${totalPages}</span>
            <button class="pagination-btn" onclick="changePage(${dashboardCurrentPage + 1})" ${dashboardCurrentPage === totalPages ? 'disabled' : ''}>Next ›</button>
            <button class="pagination-btn" onclick="changePage(${totalPages})" ${dashboardCurrentPage === totalPages ? 'disabled' : ''}>Last »</button>
        </div>`;
    }

    container.innerHTML = html;
    
    setupResultsSearch();
}

function displayResultsWithProgress(results, target, days, progressInfo) {
    const container = document.getElementById('resultsContainer');
    
    // Store current results for pagination
    currentDisplayedResults = results;
    
    const totalPages = Math.ceil(results.length / ITEMS_PER_PAGE);
    const startIndex = (dashboardCurrentPage - 1) * ITEMS_PER_PAGE;
    const endIndex = startIndex + ITEMS_PER_PAGE;
    const paginatedResults = results.slice(startIndex, endIndex);

    let html = '';

    // Show progress header if loading more
    if (progressInfo && progressInfo.isPartial) {
        const percentage = Math.round((progressInfo.loaded / progressInfo.total) * 100);
        html += `
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
        const filterInfo = selectedIndicators.length > 0 ? ` (${selectedIndicators.length} indicators)` : '';
        // html += `<div class="results-progress-header"><span class="progress-count">✅ ${results.length} signals analyzed${filterInfo}</span><span class="page-info">Page ${dashboardCurrentPage} of ${totalPages}</span></div>`;
    }

    html += `<table class="results-table"><thead><tr>
        <th class="col-no">No.</th>
        <th class="col-symbol">Company Symbol</th>
        <th class="col-indicator">Indicator</th>
        <th class="col-total">Total Signals</th>
        <th class="col-success">Success</th>
        <th class="col-failure">Failure</th>
        <th class="col-open">Open</th>
        <th class="col-rate">Success %</th>
        <th class="col-action">Action</th>
    </tr></thead><tbody>`;

    if (paginatedResults.length === 0) {
        html += '<tr><td colspan="9" class="center">No results</td></tr>';
    } else {
        // Count total signals per company across ALL results (not just current page)
        const totalSymbolCounts = {};
        results.forEach(result => {
            totalSymbolCounts[result.symbol] = (totalSymbolCounts[result.symbol] || 0) + 1;
        });
        
        // Group by symbol on current page
        const symbolGroups = {};
        paginatedResults.forEach((result, index) => {
            if (!symbolGroups[result.symbol]) {
                symbolGroups[result.symbol] = [];
            }
            symbolGroups[result.symbol].push({ result, index });
        });

        paginatedResults.forEach((result, index) => {
            const globalIndex = startIndex + index + 1;
            const totalSignals = result.totalSignals || 0;
            const successful = result.successful || 0;
            const failed = (result.completedTrades || 0) - successful;
            const openTrades = result.openTrades || 0;
            const successRate = result.successRate || 0;
            const successClass = successRate >= 70 ? 'high' : successRate >= 50 ? 'medium' : 'low';
            const hasNoData = totalSignals === 0;
            
            const symbolGroup = symbolGroups[result.symbol];
            const isMultiIndicator = symbolGroup.length > 1;
            const isFirstInGroup = isMultiIndicator && symbolGroup[0].index === index;
            const totalIndicatorCount = totalSymbolCounts[result.symbol]; // Total across all pages
            
            let rowClasses = [];
            if (hasNoData) rowClasses.push('no-data-row');
            if (isMultiIndicator) {
                rowClasses.push('company-group-member');
                if (isFirstInGroup) rowClasses.push('company-group-start');
            }
            const rowClass = rowClasses.join(' ');
            
            let indicatorDisplay = result.indicator;
            let indicatorParam = result.indicator;
            if (result.indicator === 'Short' || result.indicator === 'Long' || result.indicator === 'Standard') {
                indicatorDisplay = `MACD_${result.indicator}`;
            }
            
            const viewDetailsUrl = `/symbol/${encodeURIComponent(result.symbol)}?indicator=${encodeURIComponent(indicatorParam)}`;

            // Show total count badge if company has multiple signals
            const symbolDisplay = isFirstInGroup && totalIndicatorCount > 1
                ? `${result.symbol}<span class="indicator-count-badge">${totalIndicatorCount} signals</span>`
                : result.symbol;

            html += `<tr class="${rowClass}">
                <td class="rank">${globalIndex}</td>
                <td class="symbol">${symbolDisplay}</td>
                <td class="col-indicator"><div class="indicator-single">${indicatorDisplay}</div></td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">No Data</span>' : totalSignals}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-success">${successful}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-failure">${failed}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-open">${openTrades}</span>`}</td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">N/A</span>' : `<span class="badge badge-rate success-rate-${successClass}">${successRate}%</span>`}</td>
                <td class="center"><a href="${viewDetailsUrl}" class="btn-view" target="_blank" rel="noopener noreferrer">VIEW DETAILS</a></td>
            </tr>`;
        });
    }

    html += '</tbody></table>';

    if (results.length > ITEMS_PER_PAGE) {
        html += `<div class="pagination">
            <button class="pagination-btn" onclick="changePage(1)" ${dashboardCurrentPage === 1 ? 'disabled' : ''}>« First</button>
            <button class="pagination-btn" onclick="changePage(${dashboardCurrentPage - 1})" ${dashboardCurrentPage === 1 ? 'disabled' : ''}>‹ Prev</button>
            <span class="pagination-info">Page ${dashboardCurrentPage} of ${totalPages}</span>
            <button class="pagination-btn" onclick="changePage(${dashboardCurrentPage + 1})" ${dashboardCurrentPage === totalPages ? 'disabled' : ''}>Next ›</button>
            <button class="pagination-btn" onclick="changePage(${totalPages})" ${dashboardCurrentPage === totalPages ? 'disabled' : ''}>Last »</button>
        </div>`;
    }

    container.innerHTML = html;
    
    setupResultsSearch();
}

function changePage(page) {
    const resultsToUse = currentDisplayedResults.length > 0 ? currentDisplayedResults : allResults;
    const totalPages = Math.ceil(resultsToUse.length / ITEMS_PER_PAGE);
    if (page < 1 || page > totalPages) return;
    dashboardCurrentPage = page;
    displayResults(resultsToUse, null, null);
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Indicator filtering (same as progressive.js)
function setupIndicatorFiltering() {
    const listElement = document.getElementById('indicatorsList');
    if (!listElement) return;
    
    listElement.addEventListener('click', function(e) {
        const badge = e.target.closest('.indicator-badge');
        if (!badge) return;
        
        const indicatorName = badge.textContent.trim();
        
        if (badge.classList.contains('selected')) {
            badge.classList.remove('selected');
            selectedIndicators = selectedIndicators.filter(i => i !== indicatorName);
        } else {
            badge.classList.add('selected');
            if (!selectedIndicators.includes(indicatorName)) {
                selectedIndicators.push(indicatorName);
            }
        }
        
        const countElement = document.getElementById('indicatorCount');
        if (countElement) {
            const totalIndicators = document.querySelectorAll('.indicator-badge').length;
            const selectedCount = selectedIndicators.length;
            if (selectedCount > 0) {
                countElement.textContent = `${selectedCount}/${totalIndicators}`;
                countElement.style.background = '#f59e0b';
            } else {
                countElement.textContent = `${totalIndicators}`;
                countElement.style.background = '#3b82f6';
            }
        }
        
        if (selectedIndicators.length > 0) {
            showNotification(`${selectedIndicators.length} indicator(s) selected. Click ANALYZE to see results.`, 'info');
        }
    });
}

// Smart search - searches ALL results across all pages
let searchFilteredResults = [];
let isSearchActive = false;
let currentMinSignals = 1; // Track current filter setting

function setupResultsSearch() {
    const searchInput = document.getElementById('resultsSearch');
    const clearBtn = document.getElementById('clearSearch');
    
    if (!searchInput) return;
    
    // Initialize button state
    if (clearBtn) {
        clearBtn.style.display = searchInput.value.trim() ? 'flex' : 'none';
    }
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase().trim();
        
        if (clearBtn) {
            clearBtn.style.display = searchTerm ? 'flex' : 'none';
        }
        
        if (!searchTerm) {
            // No search term - apply min signals filter if active
            isSearchActive = false;
            searchFilteredResults = [];
            dashboardCurrentPage = 1;
            
            // Apply min signals filter
            let filteredResults = [...allResults];
            if (currentMinSignals > 1) {
                filteredResults = applyMinSignalsFilter(filteredResults, currentMinSignals);
            }
            
            displayResults(filteredResults, null, null);
            return;
        }
        
        // Search across ALL results (not just current page)
        isSearchActive = true;
        
        // First apply min signals filter, then search
        let baseResults = [...allResults];
        if (currentMinSignals > 1) {
            baseResults = applyMinSignalsFilter(baseResults, currentMinSignals);
        }
        
        searchFilteredResults = baseResults.filter(result => {
            const symbol = result.symbol.toLowerCase();
            let indicator = result.indicator.toLowerCase();
            
            // Handle MACD naming
            if (indicator === 'short' || indicator === 'long' || indicator === 'standard') {
                indicator = `macd_${indicator}`;
            }
            
            return symbol.includes(searchTerm) || indicator.includes(searchTerm);
        });
        
        // Reset to page 1 and display filtered results
        dashboardCurrentPage = 1;
        displaySearchResults(searchFilteredResults, searchTerm);
    });
    
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            searchInput.value = '';
            isSearchActive = false;
            searchFilteredResults = [];
            dashboardCurrentPage = 1;
            
            // Apply min signals filter when clearing search
            let filteredResults = [...allResults];
            if (currentMinSignals > 1) {
                filteredResults = applyMinSignalsFilter(filteredResults, currentMinSignals);
            }
            
            displayResults(filteredResults, null, null);
            searchInput.focus();
        });
    }
}

function displaySearchResults(results, searchTerm) {
    const container = document.getElementById('resultsContainer');
    
    const totalPages = Math.ceil(results.length / ITEMS_PER_PAGE);
    const startIndex = (dashboardCurrentPage - 1) * ITEMS_PER_PAGE;
    const endIndex = startIndex + ITEMS_PER_PAGE;
    const paginatedResults = results.slice(startIndex, endIndex);

    let html = '';

    // Header with search info
    html += `<div class="results-progress-header">
        <span class="progress-count">${results.length} of ${allResults.length} signals (searching: "${searchTerm}")</span>
        <span class="page-info">Page ${dashboardCurrentPage} of ${totalPages}</span>
    </div>`;

    html += `<table class="results-table"><thead><tr>
        <th class="col-no">No.</th>
        <th class="col-symbol">Company Symbol</th>
        <th class="col-indicator">Indicator</th>
        <th class="col-total">Total Signals</th>
        <th class="col-success">Success</th>
        <th class="col-failure">Failure</th>
        <th class="col-open">Open</th>
        <th class="col-rate">Success %</th>
        <th class="col-action">Action</th>
    </tr></thead><tbody>`;

    if (paginatedResults.length === 0) {
        html += `<tr><td colspan="9" class="center">No results found for "${searchTerm}"</td></tr>`;
    } else {
        // Count total signals per company across ALL search results (not just current page)
        const totalSymbolCounts = {};
        results.forEach(result => {
            totalSymbolCounts[result.symbol] = (totalSymbolCounts[result.symbol] || 0) + 1;
        });
        
        // Group by symbol on current page
        const symbolGroups = {};
        paginatedResults.forEach((result, index) => {
            if (!symbolGroups[result.symbol]) {
                symbolGroups[result.symbol] = [];
            }
            symbolGroups[result.symbol].push({ result, index });
        });

        paginatedResults.forEach((result, index) => {
            const globalIndex = startIndex + index + 1;
            const totalSignals = result.totalSignals || 0;
            const successful = result.successful || 0;
            const failed = (result.completedTrades || 0) - successful;
            const openTrades = result.openTrades || 0;
            const successRate = result.successRate || 0;
            const successClass = successRate >= 70 ? 'high' : successRate >= 50 ? 'medium' : 'low';
            const hasNoData = totalSignals === 0;
            
            const symbolGroup = symbolGroups[result.symbol];
            const isMultiIndicator = symbolGroup.length > 1;
            const isFirstInGroup = isMultiIndicator && symbolGroup[0].index === index;
            const totalIndicatorCount = totalSymbolCounts[result.symbol]; // Total across all pages
            
            let rowClasses = [];
            if (hasNoData) rowClasses.push('no-data-row');
            if (isMultiIndicator) {
                rowClasses.push('company-group-member');
                if (isFirstInGroup) rowClasses.push('company-group-start');
            }
            const rowClass = rowClasses.join(' ');
            
            let indicatorDisplay = result.indicator;
            let indicatorParam = result.indicator;
            if (result.indicator === 'Short' || result.indicator === 'Long' || result.indicator === 'Standard') {
                indicatorDisplay = `MACD_${result.indicator}`;
            }
            
            const viewDetailsUrl = `/symbol/${encodeURIComponent(result.symbol)}?indicator=${encodeURIComponent(indicatorParam)}`;

            // Show total count badge if company has multiple signals
            const symbolDisplay = isFirstInGroup && totalIndicatorCount > 1
                ? `${result.symbol}<span class="indicator-count-badge">${totalIndicatorCount} signals</span>`
                : result.symbol;

            html += `<tr class="${rowClass}">
                <td class="rank">${globalIndex}</td>
                <td class="symbol">${symbolDisplay}</td>
                <td class="col-indicator"><div class="indicator-single">${indicatorDisplay}</div></td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">No Data</span>' : totalSignals}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-success">${successful}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-failure">${failed}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-open">${openTrades}</span>`}</td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">N/A</span>' : `<span class="badge badge-rate success-rate-${successClass}">${successRate}%</span>`}</td>
                <td class="center"><a href="${viewDetailsUrl}" class="btn-view" target="_blank" rel="noopener noreferrer">VIEW DETAILS</a></td>
            </tr>`;
        });
    }

    html += '</tbody></table>';

    if (results.length > ITEMS_PER_PAGE) {
        html += `<div class="pagination">
            <button class="pagination-btn" onclick="changeSearchPage(1)" ${dashboardCurrentPage === 1 ? 'disabled' : ''}>« First</button>
            <button class="pagination-btn" onclick="changeSearchPage(${dashboardCurrentPage - 1})" ${dashboardCurrentPage === 1 ? 'disabled' : ''}>‹ Prev</button>
            <span class="pagination-info">Page ${dashboardCurrentPage} of ${totalPages}</span>
            <button class="pagination-btn" onclick="changeSearchPage(${dashboardCurrentPage + 1})" ${dashboardCurrentPage === totalPages ? 'disabled' : ''}>Next ›</button>
            <button class="pagination-btn" onclick="changeSearchPage(${totalPages})" ${dashboardCurrentPage === totalPages ? 'disabled' : ''}>Last »</button>
        </div>`;
    }

    container.innerHTML = html;
}

function changeSearchPage(page) {
    if (isSearchActive && searchFilteredResults.length > 0) {
        const totalPages = Math.ceil(searchFilteredResults.length / ITEMS_PER_PAGE);
        if (page < 1 || page > totalPages) return;
        dashboardCurrentPage = page;
        const searchInput = document.getElementById('resultsSearch');
        const searchTerm = searchInput ? searchInput.value.trim() : '';
        displaySearchResults(searchFilteredResults, searchTerm);
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
        changePage(page);
    }
}

// CLIENT-SIDE filter: Apply minimum signals filter to already-loaded results
function applyMinSignalsFilterToResults() {
    const minSignals = parseInt(document.getElementById('minSignals').value) || 1;
    currentMinSignals = minSignals; // Track current filter
    
    console.log(`[FILTER] Applying filter: ${minSignals}+ signals`);
    console.log(`[FILTER] Total results in allResults: ${allResults ? allResults.length : 0}`);
    
    if (!allResults || allResults.length === 0) {
        console.log('[FILTER] No results to filter');
        return;
    }
    
    // If search is active, re-apply search with new filter
    const searchInput = document.getElementById('resultsSearch');
    if (searchInput && searchInput.value.trim()) {
        console.log('[FILTER] Search is active, re-applying search');
        // Trigger search input event to re-filter with new min signals
        searchInput.dispatchEvent(new Event('input'));
        return;
    }
    
    // No search active - just apply min signals filter
    let filteredResults = [...allResults];
    
    console.log(`[FILTER] Before filter: ${filteredResults.length} results`);
    
    // Apply minimum signals filter
    if (minSignals > 1) {
        filteredResults = applyMinSignalsFilter(filteredResults, minSignals);
    }
    
    console.log(`[FILTER] After filter: ${filteredResults.length} results`);
    
    // Count unique companies
    const symbolCounts = {};
    filteredResults.forEach(r => {
        symbolCounts[r.symbol] = (symbolCounts[r.symbol] || 0) + 1;
    });
    console.log('[FILTER] Symbol counts:', symbolCounts);
    
    // Re-display with filter applied
    const target = document.getElementById('dashboardTarget').value;
    const days = document.getElementById('dashboardDays').value;
    dashboardCurrentPage = 1; // Reset to page 1
    displayResults(filteredResults, target, days);
    
    // Show notification
    const symbolCount = Object.keys(symbolCounts).length;
    if (minSignals > 1) {
        showNotification(`Filtered to ${symbolCount} companies with ${minSignals}+ signals`, 'info');
    } else {
        showNotification(`Showing all ${symbolCount} companies`, 'info');
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        setupIndicatorFiltering();
        setupResultsSearch();
        setupGlobalSearch();
        
        // Setup min signals dropdown listener
        const minSignalsDropdown = document.getElementById('minSignals');
        if (minSignalsDropdown) {
            minSignalsDropdown.addEventListener('change', applyMinSignalsFilterToResults);
        }
    });
} else {
    setupIndicatorFiltering();
    setupResultsSearch();
    setupGlobalSearch();
    
    // Setup min signals dropdown listener
    const minSignalsDropdown = document.getElementById('minSignals');
    if (minSignalsDropdown) {
        minSignalsDropdown.addEventListener('change', applyMinSignalsFilterToResults);
    }
}

// =========================================================
// GLOBAL COMPANY SEARCH
// =========================================================
let globalSearchTimeout = null;

function setupGlobalSearch() {
    const searchInput = document.getElementById('globalSearch');
    const clearBtn = document.getElementById('clearGlobalSearch');
    const resultsContainer = document.getElementById('globalSearchResults');
    
    if (!searchInput || !resultsContainer) return;
    
    // Initialize button state
    if (clearBtn) {
        clearBtn.style.display = searchInput.value.trim() ? 'flex' : 'none';
    }
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.trim();
        
        // Show/hide clear button
        if (clearBtn) {
            clearBtn.style.display = searchTerm ? 'flex' : 'none';
        }
        
        // Clear previous timeout
        if (globalSearchTimeout) {
            clearTimeout(globalSearchTimeout);
        }
        
        if (!searchTerm) {
            resultsContainer.classList.add('hidden');
            resultsContainer.innerHTML = '';
            return;
        }
        
        // Show loading state
        resultsContainer.classList.remove('hidden');
        resultsContainer.innerHTML = '<div class="global-search-loading">Searching...</div>';
        
        // Debounce search (wait 300ms after user stops typing)
        globalSearchTimeout = setTimeout(() => {
            performGlobalSearch(searchTerm, resultsContainer);
        }, 300);
    });
    
    // Clear button
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            searchInput.value = '';
            resultsContainer.classList.add('hidden');
            resultsContainer.innerHTML = '';
            clearBtn.style.display = 'none';
            searchInput.focus();
        });
    }
    
    // Close results when clicking outside
    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !resultsContainer.contains(e.target)) {
            resultsContainer.classList.add('hidden');
        }
    });
    
    // Reopen results when clicking on input (if there's a search term)
    searchInput.addEventListener('click', function() {
        if (this.value.trim() && resultsContainer.innerHTML) {
            resultsContainer.classList.remove('hidden');
        }
    });
}

async function performGlobalSearch(searchTerm, resultsContainer) {
    try {
        const response = await fetch(`/api/symbols?q=${encodeURIComponent(searchTerm)}`);
        const symbols = await response.json();
        
        if (symbols.length === 0) {
            resultsContainer.innerHTML = '<div class="global-search-no-results">No companies found</div>';
            return;
        }
        
        // Display results
        let html = '';
        symbols.forEach(symbol => {
            html += `
                <div class="global-search-result-item" onclick="navigateToSymbol('${symbol}')">
                    <span class="symbol-text">${symbol}</span>
                </div>
            `;
        });
        
        resultsContainer.innerHTML = html;
        
    } catch (error) {
        console.error('Global search error:', error);
        resultsContainer.innerHTML = '<div class="global-search-no-results">Search failed</div>';
    }
}

function navigateToSymbol(symbol) {
    // Clear the search input and hide results
    const searchInput = document.getElementById('globalSearch');
    const clearBtn = document.getElementById('clearGlobalSearch');
    const resultsContainer = document.getElementById('globalSearchResults');
    
    if (searchInput) {
        searchInput.value = '';
    }
    
    if (clearBtn) {
        clearBtn.style.display = 'none';
    }
    
    if (resultsContainer) {
        resultsContainer.classList.add('hidden');
        resultsContainer.innerHTML = '';
    }
    
    // Navigate to symbol page
    window.open(`/symbol/${encodeURIComponent(symbol)}`, '_blank');
}

// Expose function for button handler (used by script.js)
window.analyzeDashboardProgressive = analyzeDashboardFast;


// =========================================================
// GROUPED ANALYSIS - BY COMPANY
// =========================================================
async function analyzeDashboardGrouped() {
    const target = document.getElementById('dashboardTarget').value;
    const days = document.getElementById('dashboardDays').value;

    if (!target || !days) {
        showNotification('Fill target and days', 'warning');
        return;
    }

    try {
        allResults = [];
        dashboardCurrentPage = 1;

        document.getElementById('loadingState').classList.remove('hidden');
        document.getElementById('resultsSection').classList.add('hidden');

        console.log('📊 [DASHBOARD] Starting GROUPED analysis...');
        const startTime = performance.now();

        // Call grouped endpoint
        const url = `/api/analyze-grouped?target=${target}&days=${days}`;
        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            showError('Analysis failed: ' + data.error);
            document.getElementById('loadingState').classList.add('hidden');
            return;
        }

        allResults = data.results || [];
        
        console.log(`✅ [DASHBOARD] Grouped analysis complete: ${allResults.length} companies`);
        
        // Show results
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');
        
        displayGroupedResults(allResults, target, days);

        const endTime = performance.now();
        const totalTime = ((endTime - startTime) / 1000).toFixed(2);
        console.log(`⏱️ [DASHBOARD] Total time: ${totalTime}s`);

    } catch (error) {
        console.error('❌ [DASHBOARD] Error:', error);
        showError('Analysis failed: ' + error.message);
        document.getElementById('loadingState').classList.add('hidden');
    }
}

// Display grouped results (one row per company)
function displayGroupedResults(results, target, days) {
    const container = document.getElementById('resultsContainer');
    
    if (!results || results.length === 0) {
        container.innerHTML = '<div class="empty-state">No results found</div>';
        return;
    }

    let html = `
        <table class="results-table">
            <thead>
                <tr>
                    <th>NO.</th>
                    <th>COMPANY SYMBOL</th>
                    <th>INDICATORS</th>
                    <th>TOTAL SIGNALS</th>
                    <th>SUCCESS</th>
                    <th>FAILURE</th>
                    <th>OPEN</th>
                    <th>SUCCESS %</th>
                    <th>ACTION</th>
                </tr>
            </thead>
            <tbody>
    `;

    results.forEach((result, index) => {
        const successClass = result.successRate >= 50 ? 'success-high' : 'success-low';
        const indicatorBadge = result.indicator_count > 1 ? 
            `<span class="badge badge-power">${result.indicator_count} signals</span>` : '';
        
        html += `
            <tr>
                <td>${index + 1}</td>
                <td>
                    <strong>${result.symbol}</strong>
                    ${indicatorBadge}
                </td>
                <td><small>${result.indicators}</small></td>
                <td>${result.totalSignals}</td>
                <td class="success-cell">${result.successful}</td>
                <td class="failure-cell">${result.failed}</td>
                <td class="open-cell">${result.open}</td>
                <td class="${successClass}">${result.successRate}%</td>
                <td>
                    <a href="/symbol/${result.symbol}" class="btn btn-sm btn-primary" target="_blank">
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
