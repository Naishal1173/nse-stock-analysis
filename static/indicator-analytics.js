// Indicator Analytics Page - JavaScript
let iaData = null;
let iaFilteredData = null;
let expandedRows = new Set();

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
        document.getElementById('iaFilterBar').classList.add('hidden');
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

        document.getElementById('iaFilterBar').classList.remove('hidden');
        document.getElementById('iaResults').classList.remove('hidden');

        showNotification(
            `✅ Analyzed ${data.total_signals} signals across ${data.indicators.length} indicators in ${data.processing_time_seconds}s`,
            'success'
        );

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
                <tr class="ia-company-header-row">
                    <td colspan="9">
                        <div class="ia-company-section">
                            <div class="ia-company-title">
                                Companies with <strong>${ind.displayName}</strong> buy signals
                                <span class="ia-company-total">${ind.uniqueCompanies} companies</span>
                            </div>
                            <table class="ia-company-table">
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
                    <tr class="ia-company-row">
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

// Search functionality
document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('iaSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function () {
            const term = this.value.toLowerCase().trim();
            if (!iaData) return;

            if (!term) {
                iaFilteredData = [...iaData.indicators];
            } else {
                iaFilteredData = iaData.indicators.filter(ind => {
                    // Search by indicator name
                    if (ind.displayName.toLowerCase().includes(term)) return true;
                    if (ind.indicator.toLowerCase().includes(term)) return true;
                    // Also search by company name within this indicator
                    if (ind.companies.some(c => c.symbol.toLowerCase().includes(term))) return true;
                    return false;
                });
            }

            renderTable(iaFilteredData);
        });
    }
});

