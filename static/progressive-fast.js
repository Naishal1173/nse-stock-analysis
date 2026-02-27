// FAST VERSION - Loads all results at once
let allResults = [];
let ungroupedResults = []; // Store original ungrouped results for fast toggling
let currentDisplayedResults = []; // Track what's currently being displayed
let dashboardCurrentPage = 1;
const ITEMS_PER_PAGE = 50;
let selectedIndicators = [];
let latestPrices = {}; // Store latest prices for all symbols

async function analyzeDashboardFast() {
    const target = document.getElementById('dashboardTarget').value;
    const days = document.getElementById('dashboardDays').value;

    if (!target || !days) {
        showNotification('Fill target and days', 'warning');
        return;
    }

    // Check if data already exists and indicators are selected
    if (allResults && allResults.length > 0 && selectedIndicators.length > 0) {
        // Data already loaded - just filter the existing data
        console.log('📊 [DASHBOARD] Data already loaded, filtering locally...');
        filterBySelectedIndicators();
        return;
    }

    try {
        allResults = [];
        dashboardCurrentPage = 1;

        document.getElementById('loadingState').classList.remove('hidden');
        document.getElementById('resultsSection').classList.add('hidden');

        console.log('📊 [DASHBOARD] Starting PROGRESSIVE analysis...');
        const startTime = performance.now();

        // Build URL with optional indicator filter
        let firstUrl = `/api/analyze-progressive?target=${target}&days=${days}&batch_size=50&offset=0`;
        
        // Add indicator filter if indicators are selected
        if (selectedIndicators.length > 0) {
            const indicators = selectedIndicators.map(ind => {
                // Convert display names back to API names
                if (ind.startsWith('MACD_')) return ind.replace('MACD_', '');
                return ind;
            }).join(',');
            firstUrl += `&indicators=${encodeURIComponent(indicators)}`;
            console.log('📊 [DASHBOARD] Fetching with indicator filter:', indicators);
        }

        // STEP 1: Get first 50 results (backend analyzes ALL, returns first 50)
        console.log('📊 [DASHBOARD] Loading first 50 results...');
        const firstResponse = await fetch(firstUrl);
        const firstData = await firstResponse.json();

        if (firstData.error) {
            showError('Analysis failed: ' + firstData.error);
            document.getElementById('loadingState').classList.add('hidden');
            return;
        }

        allResults = firstData.results || [];
        const totalSignals = firstData.total_signals;

        // Show cache status
        if (firstData.cached) {
            console.log(`✅ [DASHBOARD] First ${allResults.length} results loaded from cache in ${firstData.processing_time_seconds}s (⚡ cache age: ${firstData.cache_age_seconds}s)`);
        } else {
            console.log(`✅ [DASHBOARD] First ${allResults.length} results loaded in ${firstData.processing_time_seconds}s`);
        }
        
        // Show first 50 immediately
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');
        
        let filteredResults = applySmartSorting(allResults);
        
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
            let remainingUrl = `/api/analyze-progressive?target=${target}&days=${days}&batch_size=${remainingSize}&offset=50`;
            
            // Add indicator filter to remaining request too
            if (selectedIndicators.length > 0) {
                const indicators = selectedIndicators.map(ind => {
                    if (ind.startsWith('MACD_')) return ind.replace('MACD_', '');
                    return ind;
                }).join(',');
                remainingUrl += `&indicators=${encodeURIComponent(indicators)}`;
            }
            
            const remainingResponse = await fetch(remainingUrl);
            const remainingData = await remainingResponse.json();

            if (!remainingData.error && remainingData.results) {
                allResults = allResults.concat(remainingData.results);
                console.log(`✅ [DASHBOARD] All ${allResults.length} results loaded!`);
                
                // Apply sorting to all results
                filteredResults = applySmartSorting(allResults);
                allResults = filteredResults;
                
                // Store ungrouped results for fast toggling
                ungroupedResults = [...allResults];
                
                // Final display
                displayResults(allResults, target, days);
                
                // Load prices AFTER data is loaded (only for companies with signals)
                loadLatestPrices();
                
                // Enable price filter after first successful analysis
                enablePriceFilter();
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

function filterExistingDashboardData(selectedFilterOptions) {
    if (!allResults || allResults.length === 0) {
        console.log('[DASHBOARD] No data to filter');
        return;
    }

    // If "all" is selected or nothing selected, show all
    if (selectedFilterOptions.includes('all') || selectedFilterOptions.length === 0) {
        displayResults(allResults, null, null);
        showNotification(`Showing all ${allResults.length} signals`, 'success');
        return;
    }

    // Filter to only selected indicators
    const filteredResults = allResults.filter(result => {
        let resultIndicator = result.indicator;
        
        // Handle MACD naming
        if (resultIndicator === 'Short' || resultIndicator === 'Long' || resultIndicator === 'Standard') {
            resultIndicator = `MACD_${resultIndicator}`;
        }
        
        // Check if this indicator is in the selected list
        return selectedFilterOptions.some(sel => {
            // Handle MACD_ prefix in selection
            if (sel.startsWith('MACD_')) {
                return resultIndicator === sel;
            }
            return resultIndicator === sel || result.indicator === sel;
        });
    });

    // Re-render with filtered data
    dashboardCurrentPage = 1; // Reset to page 1
    displayResults(filteredResults, null, null);
    
    showNotification(`Filtered to ${filteredResults.length} signal(s) from ${selectedFilterOptions.length} indicator(s)`, 'success');
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

    // For ungrouped view: show count of signal rows, not sum of totalSignals
    // Each row represents one indicator signal for one company
    
    // Add total signals header with pagination
    const filterInfo = selectedIndicators.length > 0 ? ` (${selectedIndicators.length} indicators)` : '';
    html += `<div class="results-summary-bar">
        <span class="total-signals-count">📊 Total: <strong>${results.length}</strong> signals${filterInfo}</span>
        <span class="page-info">Page <strong>${dashboardCurrentPage}</strong> of <strong>${totalPages}</strong></span>
    </div>`;

    html += `<table class="results-table"><thead><tr>
        <th class="col-no">NO.</th>
        <th class="col-symbol">COMPANY</th>
        <th class="col-indicator">INDICATOR</th>
        <th class="col-total center">TOTAL</th>
        <th class="col-success center">SUCCESS</th>
        <th class="col-failure center">FAILURE</th>
        <th class="col-open center">OPEN</th>
        <th class="col-rate center">SUCCESS %</th>
        <th class="col-action center">ACTION</th>
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
                ? `${result.symbol}<span class="indicator-count-badge">${totalIndicatorCount} indicators</span>`
                : result.symbol;
            
            // Success rate badge class
            let rateClass = "badge-rate";
            if (successRate >= 60) rateClass = "success-rate-high";
            else if (successRate >= 40) rateClass = "success-rate-medium";
            else rateClass = "success-rate-low";

            html += `<tr class="${rowClass}">
                <td class="rank center">${globalIndex}</td>
                <td class="symbol">${symbolDisplay}</td>
                <td class="indicator">${indicatorDisplay}</td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">No Data</span>' : totalSignals}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-success">${successful}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-failure">${failed}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-open">${openTrades}</span>`}</td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">N/A</span>' : `<span class="badge badge-rate ${rateClass}">${successRate}%</span>`}</td>
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
        // Show summary bar when loading is complete
        const filterInfo = selectedIndicators.length > 0 ? ` (${selectedIndicators.length} indicators)` : '';
        html += `<div class="results-summary-bar">
            <span class="total-signals-count">📊 Total: <strong>${results.length}</strong> signals${filterInfo}</span>
            <span class="page-info">Page <strong>${dashboardCurrentPage}</strong> of <strong>${totalPages}</strong></span>
        </div>`;
    }

    html += `<table class="results-table"><thead><tr>
        <th class="col-no">NO.</th>
        <th class="col-symbol">COMPANY</th>
        <th class="col-indicator">INDICATOR</th>
        <th class="col-total center">TOTAL</th>
        <th class="col-success center">SUCCESS</th>
        <th class="col-failure center">FAILURE</th>
        <th class="col-open center">OPEN</th>
        <th class="col-rate center">SUCCESS %</th>
        <th class="col-action center">ACTION</th>
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
                ? `${result.symbol}<span class="indicator-count-badge">${totalIndicatorCount} indicators</span>`
                : result.symbol;
            
            // Success rate badge class
            let rateClass = "badge-rate";
            if (successRate >= 60) rateClass = "success-rate-high";
            else if (successRate >= 40) rateClass = "success-rate-medium";
            else rateClass = "success-rate-low";

            html += `<tr class="${rowClass}">
                <td class="rank center">${globalIndex}</td>
                <td class="symbol">${symbolDisplay}</td>
                <td class="indicator">${indicatorDisplay}</td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">No Data</span>' : totalSignals}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-success">${successful}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-failure">${failed}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-open">${openTrades}</span>`}</td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">N/A</span>' : `<span class="badge badge-rate ${rateClass}">${successRate}%</span>`}</td>
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
        // Handle Select All button
        if (e.target.id === 'btnSelectAllIndicators' || e.target.closest('#btnSelectAllIndicators')) {
            e.stopPropagation();
            selectAllIndicators();
            return;
        }
        
        // Handle Clear All button
        if (e.target.id === 'btnClearAllIndicators' || e.target.closest('#btnClearAllIndicators')) {
            e.stopPropagation();
            clearAllIndicators();
            return;
        }
        
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
        
        updateIndicatorCount();
        
        // Smart filtering: If data already loaded, filter immediately
        if (allResults && allResults.length > 0) {
            console.log('[INDICATOR] Data already loaded, filtering immediately...');
            applyAllFilters(); // Use applyAllFilters to include price filter
        } else {
            // No data yet - just show notification
            if (selectedIndicators.length > 0) {
                showNotification(`${selectedIndicators.length} indicator(s) selected. Click ANALYZE to see results.`, 'info');
            }
        }
    });
}

// Select all indicators
function selectAllIndicators() {
    const badges = document.querySelectorAll('.indicator-badge');
    selectedIndicators = [];
    
    badges.forEach(badge => {
        badge.classList.add('selected');
        const indicatorName = badge.textContent.trim();
        if (!selectedIndicators.includes(indicatorName)) {
            selectedIndicators.push(indicatorName);
        }
    });
    
    updateIndicatorCount();
    
    // If data already loaded, filter immediately
    if (allResults && allResults.length > 0) {
        console.log('[INDICATOR] Selecting all indicators, filtering immediately...');
        applyAllFilters(); // Use applyAllFilters to include price filter
    } else {
        showNotification(`All ${selectedIndicators.length} indicators selected. Click ANALYZE to see results.`, 'info');
    }
}

// Clear all indicator selections
function clearAllIndicators() {
    const badges = document.querySelectorAll('.indicator-badge');
    selectedIndicators = [];
    
    badges.forEach(badge => {
        badge.classList.remove('selected');
    });
    
    updateIndicatorCount();
    
    // If data already loaded, show all results
    if (allResults && allResults.length > 0) {
        console.log('[INDICATOR] Clearing all indicators, showing all results...');
        applyAllFilters(); // Use applyAllFilters to include price filter
    } else {
        showNotification('All selections cleared', 'info');
    }
}

// Update indicator count badge
function updateIndicatorCount() {
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
}

// Filter existing results by selected indicators
function filterBySelectedIndicators() {
    if (!allResults || allResults.length === 0) {
        console.log('[INDICATOR] No data to filter');
        return;
    }

    let filteredResults;
    
    // Check if we're in grouped mode
    const groupCheckbox = document.getElementById('groupByCompany');
    const isGrouped = groupCheckbox && groupCheckbox.checked;
    
    // Determine which data source to use
    const sourceData = isGrouped ? ungroupedResults : allResults;
    
    if (!sourceData || sourceData.length === 0) {
        console.log('[INDICATOR] No source data to filter');
        return;
    }
    
    // If no indicators selected, show all
    if (selectedIndicators.length === 0) {
        filteredResults = [...sourceData];
        showNotification(`Showing all ${filteredResults.length} signals`, 'success');
    } else {
        // Filter to only selected indicators
        filteredResults = sourceData.filter(result => {
            let resultIndicator = result.indicator;
            
            // Handle MACD naming
            if (resultIndicator === 'Short' || resultIndicator === 'Long' || resultIndicator === 'Standard') {
                resultIndicator = `MACD_${resultIndicator}`;
            }
            
            return selectedIndicators.includes(resultIndicator);
        });
        
        showNotification(`Filtered to ${filteredResults.length} signal(s) from ${selectedIndicators.length} indicator(s)`, 'success');
    }

    // Apply smart sorting to filtered results
    filteredResults = applySmartSorting(filteredResults);

    // Re-render with filtered and sorted data based on view mode
    dashboardCurrentPage = 1; // Reset to page 1
    
    if (isGrouped) {
        // Group the filtered results and display
        const groupedResults = groupResultsByCompany(filteredResults);
        allResults = groupedResults; // Update allResults with grouped data
        const target = document.getElementById('dashboardTarget').value;
        const days = document.getElementById('dashboardDays').value;
        displayGroupedResults(groupedResults, target, days, filteredResults.length);
    } else {
        // Display ungrouped results
        allResults = filteredResults; // Update allResults
        displayResults(filteredResults, null, null);
    }
}

// Smart search - searches ALL results across all pages
let searchFilteredResults = [];
let isSearchActive = false;
let currentMinSignals = 1; // Track current filter setting

let searchSetupDone = false;

function setupResultsSearch() {
    const searchInput = document.getElementById('resultsSearch');
    const clearBtn = document.getElementById('clearSearch');
    
    if (!searchInput) return;
    
    // Initialize button state
    if (clearBtn) {
        if (searchInput.value.trim()) {
            clearBtn.classList.add('visible');
        } else {
            clearBtn.classList.remove('visible');
        }
    }
    
    // Prevent duplicate event listener setup
    if (searchSetupDone) return;
    searchSetupDone = true;
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase().trim();
        
        if (clearBtn) {
            if (searchTerm) {
                clearBtn.classList.add('visible');
            } else {
                clearBtn.classList.remove('visible');
            }
        }
        
        if (!searchTerm) {
            // No search term - apply min signals filter if active
            isSearchActive = false;
            searchFilteredResults = [];
            dashboardCurrentPage = 1;
            
            // Check if we're in grouped mode
            const isGrouped = allResults.length > 0 && allResults[0].indicators !== undefined;
            
            // Apply min signals filter (only for ungrouped mode)
            let filteredResults = [...allResults];
            if (currentMinSignals > 1 && !isGrouped) {
                filteredResults = applyMinSignalsFilter(filteredResults, currentMinSignals);
            }
            
            // Use appropriate display function based on mode
            if (isGrouped) {
                const target = document.getElementById('dashboardTarget').value;
                const days = document.getElementById('dashboardDays').value;
                // Pass original ungrouped count for grouped view
                displayGroupedResults(filteredResults, target, days, ungroupedResults ? ungroupedResults.length : filteredResults.length);
            } else {
                displayResults(filteredResults, null, null);
            }
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
            
            // Handle both grouped and ungrouped results
            let indicator = '';
            if (result.indicator) {
                // Ungrouped result - single indicator
                indicator = result.indicator.toLowerCase();
                // Handle MACD naming
                if (indicator === 'short' || indicator === 'long' || indicator === 'standard') {
                    indicator = `macd_${indicator}`;
                }
            } else if (result.indicators) {
                // Grouped result - multiple indicators as string
                indicator = result.indicators.toLowerCase();
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
            
            // Hide clear button
            clearBtn.classList.remove('visible');
            
            // Check if we're in grouped mode
            const isGrouped = allResults.length > 0 && allResults[0].indicators !== undefined;
            
            // Apply min signals filter when clearing search
            let filteredResults = [...allResults];
            if (currentMinSignals > 1 && !isGrouped) {
                filteredResults = applyMinSignalsFilter(filteredResults, currentMinSignals);
            }
            
            // Use appropriate display function
            if (isGrouped) {
                const target = document.getElementById('dashboardTarget').value;
                const days = document.getElementById('dashboardDays').value;
                // Pass original ungrouped count for grouped view
                displayGroupedResults(filteredResults, target, days, ungroupedResults ? ungroupedResults.length : filteredResults.length);
            } else {
                displayResults(filteredResults, null, null);
            }
            
            searchInput.focus();
        });
    }
}

function displaySearchResults(results, searchTerm) {
    const container = document.getElementById('resultsContainer');
    
    // Check if we're displaying grouped results
    const isGrouped = results.length > 0 && results[0].indicators !== undefined;
    
    if (isGrouped) {
        // Use grouped display for grouped results
        displaySearchResultsGrouped(results, searchTerm);
        return;
    }
    
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
                ? `${result.symbol}<span class="indicator-count-badge">${totalIndicatorCount} indicators</span>`
                : result.symbol;
            
            // Success rate badge class
            let rateClass = "badge-rate";
            if (successRate >= 60) rateClass = "success-rate-high";
            else if (successRate >= 40) rateClass = "success-rate-medium";
            else rateClass = "success-rate-low";

            html += `<tr class="${rowClass}">
                <td class="rank center">${globalIndex}</td>
                <td class="symbol">${symbolDisplay}</td>
                <td class="indicator">${indicatorDisplay}</td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">No Data</span>' : totalSignals}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-success">${successful}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-failure">${failed}</span>`}</td>
                <td class="center">${hasNoData ? '-' : `<span class="badge badge-open">${openTrades}</span>`}</td>
                <td class="center">${hasNoData ? '<span class="badge badge-warning">N/A</span>' : `<span class="badge badge-rate ${rateClass}">${successRate}%</span>`}</td>
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

// Display search results for grouped data
function displaySearchResultsGrouped(results, searchTerm) {
    const container = document.getElementById('resultsContainer');
    
    if (!results || results.length === 0) {
        container.innerHTML = `<div class="empty-state">No results found for "${searchTerm}"</div>`;
        return;
    }
    
    // Get target and days for URL parameters
    const target = document.getElementById('dashboardTarget').value || '5';
    const days = document.getElementById('dashboardDays').value || '30';

    let html = `
        <div class="results-progress-header">
            <span class="progress-count">${results.length} of ${allResults.length} companies (searching: "${searchTerm}")</span>
        </div>
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
        
        // Build URL with first indicator, target, and days
        const viewDetailsUrl = `/symbol/${result.symbol}?indicator=${encodeURIComponent(result.firstIndicator)}&target=${target}&days=${days}`;
        
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
                    <a href="${viewDetailsUrl}" class="btn btn-sm btn-primary" target="_blank">
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

// =========================================================
// PRICE FILTERING
// =========================================================
async function loadLatestPrices() {
    try {
        console.log('📊 [PRICES] Loading latest prices...');
        const priceFilter = document.getElementById('priceFilter');
        if (priceFilter) {
            priceFilter.disabled = true;
            priceFilter.style.opacity = '0.5';
        }
        
        // Get unique symbols from ungroupedResults (only companies with signals)
        let symbolsToFetch = [];
        if (ungroupedResults && ungroupedResults.length > 0) {
            const uniqueSymbols = [...new Set(ungroupedResults.map(r => r.symbol))];
            symbolsToFetch = uniqueSymbols;
            console.log(`📊 [PRICES] Fetching prices for ${symbolsToFetch.length} companies with signals`);
        } else {
            console.log('⚠️ [PRICES] No data loaded yet, will fetch prices after data loads');
            if (priceFilter) {
                priceFilter.disabled = false;
                priceFilter.style.opacity = '1';
            }
            return; // Don't fetch prices if no data loaded yet
        }
        
        // Build URL with symbols parameter
        let url = '/api/latest-prices';
        if (symbolsToFetch.length > 0) {
            url += `?symbols=${encodeURIComponent(symbolsToFetch.join(','))}`;
        }
        
        const response = await fetch(url);
        latestPrices = await response.json();
        console.log(`✅ [PRICES] Loaded prices for ${Object.keys(latestPrices).length} symbols`);
        
        if (priceFilter) {
            priceFilter.disabled = false;
            priceFilter.style.opacity = '1';
        }
        
        // Show notification
        showNotification(`Prices loaded for ${Object.keys(latestPrices).length} companies`, 'success');
    } catch (error) {
        console.error('❌ [PRICES] Failed to load prices:', error);
        latestPrices = {};
        
        const priceFilter = document.getElementById('priceFilter');
        if (priceFilter) {
            priceFilter.disabled = false;
            priceFilter.style.opacity = '1';
        }
        
        showNotification('Failed to load prices. Price filter may not work correctly.', 'error');
    }
}

// Apply price filter to results
function applyPriceFilter(results, priceRange) {
    if (!priceRange || priceRange === 'all') {
        return results;
    }
    
    // Check if prices are loaded
    if (!latestPrices || Object.keys(latestPrices).length === 0) {
        console.warn('⚠️ [PRICE FILTER] Prices not loaded yet, cannot filter');
        showNotification('Loading prices... Please wait and try again.', 'warning');
        return results;
    }
    
    // Parse price range
    let minPrice = 0;
    let maxPrice = Infinity;
    
    if (priceRange === '0-50') {
        minPrice = 0;
        maxPrice = 50;
    } else if (priceRange === '50-100') {
        minPrice = 50;
        maxPrice = 100;
    } else if (priceRange === '100-500') {
        minPrice = 100;
        maxPrice = 500;
    } else if (priceRange === '500-1000') {
        minPrice = 500;
        maxPrice = 1000;
    } else if (priceRange === '1000+') {
        minPrice = 1000;
        maxPrice = Infinity;
    }
    
    // Filter results based on price
    const filtered = results.filter(result => {
        const priceData = latestPrices[result.symbol];
        if (!priceData) {
            // If price not available for this symbol, exclude it from filtered results
            return false;
        }
        
        const price = priceData.price;
        // FIXED: For ranges like 50-100, use <= for maxPrice to include boundary values
        if (maxPrice === Infinity) {
            return price >= minPrice;
        } else {
            return price >= minPrice && price <= maxPrice;
        }
    });
    
    console.log(`📊 [PRICE FILTER] Filtered from ${results.length} to ${filtered.length} results (${priceRange})`);
    console.log(`📊 [PRICE FILTER] Price range: ₹${minPrice} - ${maxPrice === Infinity ? '∞' : '₹' + maxPrice}`);
    return filtered;
}

// Enable price filter after first analysis
function enablePriceFilter() {
    const priceFilter = document.getElementById('priceFilter');
    if (priceFilter) {
        priceFilter.disabled = false;
        priceFilter.style.opacity = '1';
        priceFilter.style.cursor = 'pointer';
        console.log('✅ [PRICE FILTER] Enabled after first analysis');
    }
}

// Disable price filter on page load
function disablePriceFilter() {
    const priceFilter = document.getElementById('priceFilter');
    if (priceFilter) {
        priceFilter.disabled = true;
        priceFilter.style.opacity = '0.5';
        priceFilter.style.cursor = 'not-allowed';
        priceFilter.value = 'all'; // Reset to "All Prices"
        console.log('🔒 [PRICE FILTER] Disabled - Run analysis first');
    }
}

// Setup price filter dropdown
function setupPriceFilter() {
    const priceFilter = document.getElementById('priceFilter');
    if (!priceFilter) return;
    
    priceFilter.addEventListener('change', function() {
        const priceRange = this.value;
        console.log(`📊 [PRICE FILTER] Selected: ${priceRange}`);
        
        // If data is loaded, apply filter immediately
        if (ungroupedResults && ungroupedResults.length > 0) {
            applyAllFilters();
        }
    });
}

// Apply all filters (indicators + price + min signals)
function applyAllFilters() {
    if (!ungroupedResults || ungroupedResults.length === 0) {
        console.log('[FILTER] No data to filter');
        return;
    }
    
    let filteredResults = [...ungroupedResults];
    
    // 1. Apply indicator filter
    if (selectedIndicators.length > 0) {
        filteredResults = filteredResults.filter(result => {
            let resultIndicator = result.indicator;
            
            // Handle MACD naming
            if (resultIndicator === 'Short' || resultIndicator === 'Long' || resultIndicator === 'Standard') {
                resultIndicator = `MACD_${resultIndicator}`;
            }
            
            return selectedIndicators.includes(resultIndicator);
        });
    }
    
    // 2. Apply price filter
    const priceFilter = document.getElementById('priceFilter');
    if (priceFilter) {
        const priceRange = priceFilter.value;
        filteredResults = applyPriceFilter(filteredResults, priceRange);
    }
    
    // 3. Apply smart sorting
    filteredResults = applySmartSorting(filteredResults);
    
    // 4. Check if grouped mode
    const groupCheckbox = document.getElementById('groupByCompany');
    const isGrouped = groupCheckbox && groupCheckbox.checked;
    
    // Reset to page 1
    dashboardCurrentPage = 1;
    
    if (isGrouped) {
        // Group and display
        const groupedResults = groupResultsByCompany(filteredResults);
        allResults = groupedResults;
        const target = document.getElementById('dashboardTarget').value;
        const days = document.getElementById('dashboardDays').value;
        displayGroupedResults(groupedResults, target, days, filteredResults.length);
    } else {
        // Display ungrouped
        allResults = filteredResults;
        displayResults(filteredResults, null, null);
    }
    
    // Show notification
    const priceRange = priceFilter ? priceFilter.value : 'all';
    const priceLabel = priceRange === 'all' ? '' : ` (${priceRange})`;
    const indicatorLabel = selectedIndicators.length > 0 ? ` (${selectedIndicators.length} indicators)` : '';
    showNotification(`Showing ${filteredResults.length} signals${indicatorLabel}${priceLabel}`, 'success');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        setupIndicatorFiltering();
        setupResultsSearch();
        setupGlobalSearch();
        setupPriceFilter();
        disablePriceFilter(); // Disable price filter on page load
        // Don't load prices on page load - will load after data is fetched
        
        // Setup min signals dropdown listener
        const minSignalsDropdown = document.getElementById('minSignals');
        if (minSignalsDropdown) {
            minSignalsDropdown.addEventListener('change', applyMinSignalsFilterToResults);
        }
        
        // Load dashboard data (from script.js functions)
        if (typeof loadSummaryInfo !== 'undefined') loadSummaryInfo();
        if (typeof loadAvailableIndicators !== 'undefined') loadAvailableIndicators();
        if (typeof setupIndicatorsToggle !== 'undefined') setupIndicatorsToggle();
        if (typeof loadDashboardSignals !== 'undefined') loadDashboardSignals();
    });
} else {
    setupIndicatorFiltering();
    setupResultsSearch();
    setupGlobalSearch();
    setupPriceFilter();
    disablePriceFilter(); // Disable price filter on page load
    // Don't load prices on page load - will load after data is fetched
    
    // Setup min signals dropdown listener
    const minSignalsDropdown = document.getElementById('minSignals');
    if (minSignalsDropdown) {
        minSignalsDropdown.addEventListener('change', applyMinSignalsFilterToResults);
    }
    
    // Load dashboard data (from script.js functions)
    if (typeof loadSummaryInfo !== 'undefined') loadSummaryInfo();
    if (typeof loadAvailableIndicators !== 'undefined') loadAvailableIndicators();
    if (typeof setupIndicatorsToggle !== 'undefined') setupIndicatorsToggle();
    if (typeof loadDashboardSignals !== 'undefined') loadDashboardSignals();
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
        // Check if we already have ungrouped results loaded
        if (ungroupedResults && ungroupedResults.length > 0) {
            // We have ungrouped data - group it instantly (no API call needed)
            console.log('📊 [DASHBOARD] Grouping existing results...');
            const startTime = performance.now();
            
            const groupedResults = groupResultsByCompany(ungroupedResults);
            allResults = groupedResults; // Update allResults with grouped data
            
            const endTime = performance.now();
            const totalTime = ((endTime - startTime) / 1000).toFixed(2);
            console.log(`✅ [DASHBOARD] Grouped ${groupedResults.length} companies in ${totalTime}s`);
            
            // Pass the original ungrouped count
            displayGroupedResults(groupedResults, target, days, ungroupedResults.length);
            return;
        }

        // No data loaded yet - fetch from API
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

// Group ungrouped results by company (instant, no API call)
function groupResultsByCompany(ungroupedResults) {
    const grouped = {};
    
    ungroupedResults.forEach(result => {
        const symbol = result.symbol;
        
        if (!grouped[symbol]) {
            grouped[symbol] = {
                symbol: symbol,
                indicators: [],
                totalSignals: 0,
                successful: 0,
                failed: 0,
                open: 0,
                completedTrades: 0
            };
        }
        
        grouped[symbol].indicators.push(result.indicator);
        grouped[symbol].totalSignals += result.totalSignals || 0;
        grouped[symbol].successful += result.successful || 0;
        grouped[symbol].failed += result.failed || 0;
        grouped[symbol].open += result.openTrades || 0;
        grouped[symbol].completedTrades += result.completedTrades || 0;
    });
    
    // Convert to array and calculate success rate
    const results = Object.values(grouped).map(data => {
        const successRate = data.completedTrades > 0 
            ? ((data.successful / data.completedTrades) * 100).toFixed(2)
            : 0;
        
        // First indicator is the first one in the indicators array
        const firstIndicator = data.indicators[0];
        
        return {
            symbol: data.symbol,
            indicators: data.indicators.join(', '),
            firstIndicator: firstIndicator,  // Use first from array
            indicator_count: data.indicators.length,
            totalSignals: data.totalSignals,
            successful: data.successful,
            failed: data.failed,
            open: data.open,
            completedTrades: data.completedTrades,
            successRate: parseFloat(successRate)
        };
    });
    
    // Sort by success rate (highest first), then by symbol
    results.sort((a, b) => {
        if (b.successRate !== a.successRate) {
            return b.successRate - a.successRate;
        }
        return a.symbol.localeCompare(b.symbol);
    });
    
    return results;
}

// Display grouped results (one row per company)
function displayGroupedResults(results, target, days, originalCount) {
    const container = document.getElementById('resultsContainer');
    
    if (!results || results.length === 0) {
        container.innerHTML = '<div class="empty-state">No results found</div>';
        return;
    }

    // Use original ungrouped count if provided, otherwise count from ungroupedResults
    const totalSignalCount = originalCount || (ungroupedResults ? ungroupedResults.length : results.length);

    let html = `
        <!-- Summary Bar -->
        <div class="results-summary-bar">
            <div class="total-signals-count">
                📊 Total: <strong>${totalSignalCount}</strong> signals across <strong>${results.length}</strong> companies
            </div>
        </div>
        
        <table class="results-table">
            <thead>
                <tr>
                    <th class="col-no">NO.</th>
                    <th class="col-symbol">COMPANY</th>
                    <th class="col-indicator">INDICATORS</th>
                    <th class="col-total center">TOTAL</th>
                    <th class="col-success center">SUCCESS</th>
                    <th class="col-failure center">FAILURE</th>
                    <th class="col-open center">OPEN</th>
                    <th class="col-rate center">SUCCESS %</th>
                    <th class="col-action center">ACTION</th>
                </tr>
            </thead>
            <tbody>
    `;

    results.forEach((result, index) => {
        // Success rate badge class
        let rateClass = 'badge-rate';
        if (result.successRate >= 60) rateClass = 'success-rate-high';
        else if (result.successRate >= 40) rateClass = 'success-rate-medium';
        else rateClass = 'success-rate-low';
        
        const indicatorBadge = result.indicator_count > 1 ? 
            `<span class="indicator-count-badge">${result.indicator_count} indicators</span>` : '';
        
        // Build URL with first indicator, target, and days
        const viewDetailsUrl = `/symbol/${result.symbol}?indicator=${encodeURIComponent(result.firstIndicator)}&target=${target}&days=${days}`;
        
        html += `
            <tr>
                <td class="rank center">${index + 1}</td>
                <td class="symbol">
                    ${result.symbol}
                    ${indicatorBadge}
                </td>
                <td class="indicator">${result.indicators}</td>
                <td class="center">${result.totalSignals}</td>
                <td class="center"><span class="badge badge-success">${result.successful}</span></td>
                <td class="center"><span class="badge badge-failure">${result.failed}</span></td>
                <td class="center"><span class="badge badge-open">${result.open}</span></td>
                <td class="center"><span class="badge ${rateClass}">${result.successRate}%</span></td>
                <td class="center">
                    <a href="${viewDetailsUrl}" class="btn-view" target="_blank">
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

// Expose grouped analysis function globally
window.analyzeDashboardGrouped = analyzeDashboardGrouped;
window.displayResults = displayResults;
