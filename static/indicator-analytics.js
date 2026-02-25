// Indicator Analytics Page - JavaScript
let iaData = null;
let iaFilteredData = null;
let expandedRows = new Set();

// Simple notification function
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    // You can add a toast notification here if needed
}

async function analyzeIndicators() {
    const target = document.getElementById('iaTarget').value;
    const days = document.getElementById('iaDays').value;

    if (!target || !days) {
        showNotification('Please fill in target and days', 'warning');
        return;
    }

    try {
        // Show loading, hide others
        document.getElementById('iaLoadingState').classList.remove('hidden');
        document.getElementById('iaEmptyState').classList.add('hidden');
        document.getElementById('iaResults').classList.add('hidden');

        const url = `/api/indicator-analytics?target=${target}&days=${days}`;
        console.log('[IA] Fetching:', url);

        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            showNotification('Analysis failed: ' + data.error, 'error');
            document.getElementById('iaLoadingState').classList.add('hidden');
            document.getElementById('iaEmptyState').classList.remove('hidden');
            return;
        }

        iaData = data;
        iaFilteredData = [...data.indicators];
        expandedRows.clear();

        // Hide loading
        document.getElementById('iaLoadingState').classList.add('hidden');

        // Show results
        renderTable(iaFilteredData);

        document.getElementById('iaResults').classList.remove('hidden');

        // Show notification with cache status
        let message = `✅ Analyzed ${data.total_signals} signals across ${data.indicators.length} indicators`;
        if (data.cached) {
            if (data.reused_from === 'dashboard') {
                message += ` (⚡ reused from dashboard cache, age: ${data.cache_age_seconds}s)`;
            } else {
                message += ` (⚡ cached, age: ${data.cache_age_seconds}s)`;
            }
        } else {
            message += ` in ${data.processing_time_seconds}s`;
        }
        
        showNotification(message, 'success');

    } catch (error) {
        console.error('[IA] Error:', error);
        showNotification('Analysis failed: ' + error.message, 'error');
        document.getElementById('iaLoadingState').classList.add('hidden');
        document.getElementById('iaEmptyState').classList.remove('hidden');
    }
}

function renderTable(indicators) {
    const container = document.getElementById('iaTableContainer');

    if (!indicators || indicators.length === 0) {
        container.innerHTML = '<div class="empty-state">No indicators found</div>';
        return;
    }

    let html = `
        <table class="ia-table">
            <thead>
                <tr>
                    <th class="ia-col-no">NO.</th>
                    <th class="ia-col-expand"></th>
                    <th class="ia-col-indicator">INDICATOR</th>
                    <th class="ia-col-num">TOTAL SIGNALS</th>
                    <th class="ia-col-num">SUCCESS</th>
                    <th class="ia-col-num">FAILURE</th>
                    <th class="ia-col-num">OPEN</th>
                    <th class="ia-col-num">SUCCESS %</th>
                    <th class="ia-col-num">COMPANIES</th>
                </tr>
            </thead>
            <tbody>
    `;

    indicators.forEach((ind, index) => {
        const isExpanded = expandedRows.has(ind.indicator);
        const successClass = ind.successRate >= 70 ? 'high' : ind.successRate >= 50 ? 'medium' : 'low';

        html += `
            <tr class="ia-indicator-row ${isExpanded ? 'ia-expanded' : ''}" 
                onclick="toggleExpand('${ind.indicator}')" 
                data-indicator="${ind.indicator}">
                <td class="ia-rank">${index + 1}</td>
                <td class="ia-expand-icon">${isExpanded ? '▼' : '▶'}</td>
                <td class="ia-indicator-name">
                    <strong>${ind.displayName}</strong>
                </td>
                <td class="center">${ind.totalSignals}</td>
                <td class="center"><span class="badge badge-success">${ind.successful}</span></td>
                <td class="center"><span class="badge badge-failure">${ind.failed}</span></td>
                <td class="center"><span class="badge badge-open">${ind.open}</span></td>
                <td class="center">
                    <span class="badge badge-rate success-rate-${successClass}">${ind.successRate}%</span>
                </td>
                <td class="center">
                    <span class="ia-company-count">${ind.uniqueCompanies}</span>
                </td>
            </tr>
        `;

        // Expanded company details
        if (isExpanded && ind.companies && ind.companies.length > 0) {
            html += `
                <tr class="ia-company-header-row" onclick="event.stopPropagation()">
                    <td colspan="9">
                        <div class="ia-company-section">
                            <div class="ia-company-header-wrapper">
                                <div class="ia-company-title">
                                    Companies with <strong>${ind.displayName}</strong> buy signals
                                    <span class="ia-company-total">${ind.uniqueCompanies} companies</span>
                                </div>
                                <div class="ia-company-search-box">
                                    <input type="text" 
                                           class="ia-company-search" 
                                           id="companySearch_${ind.indicator}"
                                           placeholder="🔍 Search companies..." 
                                           autocomplete="off"
                                           onkeyup="filterCompanies('${ind.indicator}')">
                                    <button class="ia-clear-search" 
                                            id="clearCompanySearch_${ind.indicator}"
                                            onclick="clearCompanySearch('${ind.indicator}')"
                                            style="display: none;">✕</button>
                                </div>
                            </div>
                            <table class="ia-company-table" id="companyTable_${ind.indicator}">
                                <thead>
                                    <tr>
                                        <th>NO.</th>
                                        <th>COMPANY</th>
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

            ind.companies.forEach((comp, ci) => {
                const compSuccessClass = comp.successRate >= 70 ? 'high' : comp.successRate >= 50 ? 'medium' : 'low';
                const viewUrl = `/symbol/${encodeURIComponent(comp.symbol)}?indicator=${encodeURIComponent(ind.indicator)}`;

                html += `
                    <tr class="ia-company-row" data-symbol="${comp.symbol.toLowerCase()}">
                        <td>${ci + 1}</td>
                        <td><strong>${comp.symbol}</strong></td>
                        <td class="center">${comp.totalSignals}</td>
                        <td class="center"><span class="badge badge-success">${comp.successful}</span></td>
                        <td class="center"><span class="badge badge-failure">${comp.failed}</span></td>
                        <td class="center"><span class="badge badge-open">${comp.open}</span></td>
                        <td class="center">
                            <span class="badge badge-rate success-rate-${compSuccessClass}">${comp.successRate}%</span>
                        </td>
                        <td class="center">
                            <a href="${viewUrl}" class="btn-view" target="_blank" rel="noopener noreferrer">VIEW</a>
                        </td>
                    </tr>
                `;
            });

            html += `
                                </tbody>
                            </table>
                        </div>
                    </td>
                </tr>
            `;
        }
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function toggleExpand(indicator) {
    if (expandedRows.has(indicator)) {
        expandedRows.delete(indicator);
    } else {
        expandedRows.add(indicator);
    }
    renderTable(iaFilteredData);
}

// Close expanded sections when clicking outside
document.addEventListener('click', function(event) {
    // Check if click is outside any expanded section
    const clickedRow = event.target.closest('.ia-indicator-row');
    const clickedExpandedSection = event.target.closest('.ia-company-header-row');
    
    // If clicked outside both indicator row and expanded section, close all
    if (!clickedRow && !clickedExpandedSection && expandedRows.size > 0) {
        expandedRows.clear();
        renderTable(iaFilteredData);
    }
});

// Company search functionality
function filterCompanies(indicator) {
    const searchInput = document.getElementById(`companySearch_${indicator}`);
    const clearBtn = document.getElementById(`clearCompanySearch_${indicator}`);
    const table = document.getElementById(`companyTable_${indicator}`);
    
    if (!searchInput || !table) return;
    
    const searchTerm = searchInput.value.toLowerCase().trim();
    const rows = table.querySelectorAll('tbody tr.ia-company-row');
    
    // Show/hide clear button
    if (clearBtn) {
        clearBtn.style.display = searchTerm ? 'block' : 'none';
    }
    
    let visibleCount = 0;
    rows.forEach(row => {
        const symbol = row.getAttribute('data-symbol');
        if (!searchTerm || symbol.includes(searchTerm)) {
            row.style.display = '';
            visibleCount++;
        } else {
            row.style.display = 'none';
        }
    });
    
    // Update row numbers for visible rows
    let rowNum = 1;
    rows.forEach(row => {
        if (row.style.display !== 'none') {
            row.querySelector('td:first-child').textContent = rowNum++;
        }
    });
}

function clearCompanySearch(indicator) {
    const searchInput = document.getElementById(`companySearch_${indicator}`);
    const clearBtn = document.getElementById(`clearCompanySearch_${indicator}`);
    
    if (searchInput) {
        searchInput.value = '';
        filterCompanies(indicator);
    }
    
    if (clearBtn) {
        clearBtn.style.display = 'none';
    }
}

