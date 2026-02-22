export interface DisasterEvent {
  id: string;
  type: string;
  name: string;
  date: string;
  severity: 'Low' | 'Medium' | 'High' | 'Critical';
  region: string;
  lat: number;
  lng: number;
  displacedWorkers: number;
  mostAffectedIndustry: string;
}

export interface IndustryImpact {
  industry: string;
  riskScore: 'Low' | 'Medium' | 'High';
  recoveryTime: string;
  pivotSuggestions: string[];
  jobLossPct: number;
  demandIncreasePct: number;
  avgRecoveryMonths: number;
}

export interface DisplacementData {
  month: number;
  Hospitality: number;
  Construction: number;
  Healthcare: number;
  Retail: number;
  Insurance: number;
}

export const MOCK_DISASTERS: DisasterEvent[] = [
  {
    id: '1',
    type: 'Hurricane',
    name: 'Hurricane Ian',
    date: '2022-09-28',
    severity: 'Critical',
    region: 'Florida, US',
    lat: 27.6648,
    lng: -81.5158,
    displacedWorkers: 12400,
    mostAffectedIndustry: 'Hospitality'
  },
  {
    id: '2',
    type: 'Wildfire',
    name: 'Camp Fire',
    date: '2018-11-08',
    severity: 'High',
    region: 'California, US',
    lat: 39.7596,
    lng: -121.6219,
    displacedWorkers: 8500,
    mostAffectedIndustry: 'Retail'
  }
];

export const MOCK_INDUSTRY_IMPACT: IndustryImpact[] = [
  {
    industry: 'Hospitality',
    riskScore: 'High',
    recoveryTime: '12–18 months',
    pivotSuggestions: ['Customer support', 'Logistics', 'Healthcare admin'],
    jobLossPct: 35,
    demandIncreasePct: 5,
    avgRecoveryMonths: 15
  },
  {
    industry: 'Construction',
    riskScore: 'Low',
    recoveryTime: '3–6 months',
    pivotSuggestions: ['Infrastructure repair', 'Project management'],
    jobLossPct: 5,
    demandIncreasePct: 40,
    avgRecoveryMonths: 4
  },
  {
    industry: 'Healthcare',
    riskScore: 'Medium',
    recoveryTime: '6–9 months',
    pivotSuggestions: ['Telehealth', 'Emergency response'],
    jobLossPct: 10,
    demandIncreasePct: 25,
    avgRecoveryMonths: 8
  }
];

export const MOCK_DISPLACEMENT_CURVE: DisplacementData[] = [
  { month: -3, Hospitality: 0, Construction: 0, Healthcare: 0, Retail: 0, Insurance: 0 },
  { month: -2, Hospitality: 1, Construction: 0, Healthcare: 1, Retail: 0, Insurance: 0 },
  { month: -1, Hospitality: 0, Construction: 1, Healthcare: 0, Retail: 1, Insurance: 0 },
  { month: 0, Hospitality: -35, Construction: -5, Healthcare: -10, Retail: -15, Insurance: 2 },
  { month: 1, Hospitality: -32, Construction: 10, Healthcare: -5, Retail: -12, Insurance: 5 },
  { month: 2, Hospitality: -28, Construction: 25, Healthcare: 2, Retail: -8, Insurance: 8 },
  { month: 3, Hospitality: -22, Construction: 40, Healthcare: 8, Retail: -4, Insurance: 10 },
  { month: 6, Hospitality: -15, Construction: 30, Healthcare: 12, Retail: 2, Insurance: 5 },
  { month: 9, Hospitality: -8, Construction: 15, Healthcare: 15, Retail: 5, Insurance: 2 },
  { month: 12, Hospitality: -2, Construction: 5, Healthcare: 10, Retail: 8, Insurance: 0 },
];
